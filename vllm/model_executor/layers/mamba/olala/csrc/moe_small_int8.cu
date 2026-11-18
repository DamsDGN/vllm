// Olala latent MoE, small decode batches (M <= 8 tokens), int8 W8A16 per-channel experts, non-gated relu^2.
//   up:   router top-k (sigmoid scores ranked by score + bias, weights normalized over the k winners) per warp,
//         then h[m, k] = relu(W1[e] @ x[m] * s1[e])^2                                   (bf16, [M * K, N1])
//   down: cat[m, :D2]   = scale * sum_k w[m, k] * (W2[e] @ h[m, k]) * s2[e]               (bf16)
//         cat[m, D2:]   = relu(s[m])^2  (shared-expert pre-activation)
// Every CTA issues all its weight loads (16-byte, L1 no-allocate) before the first FMA; no fp32 partials, no
// epilogue kernel. (A single launch with the down CTAs waiting on the up CTAs measured no faster.)
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint4 ldg128(const int8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// 16 signed int8 (one uint4) dotted with 16 fp32 values; exact int8 -> fp32 via the 2^23 magic number
__device__ __forceinline__ float dot16(const uint4& q, const float* x) {
  const unsigned w[4] = {q.x ^ 0x80808080u, q.y ^ 0x80808080u, q.z ^ 0x80808080u, q.w ^ 0x80808080u};
  float a = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a = fmaf(__uint_as_float(__byte_perm(w[i], 0x4B000000u, 0x7650)) - 8388736.f, x[4 * i + 0], a);
    a = fmaf(__uint_as_float(__byte_perm(w[i], 0x4B000000u, 0x7651)) - 8388736.f, x[4 * i + 1], a);
    a = fmaf(__uint_as_float(__byte_perm(w[i], 0x4B000000u, 0x7652)) - 8388736.f, x[4 * i + 2], a);
    a = fmaf(__uint_as_float(__byte_perm(w[i], 0x4B000000u, 0x7653)) - 8388736.f, x[4 * i + 3], a);
  }
  return a;
}

__device__ __forceinline__ void load8_bf16(const __nv_bfloat16* p, float* out) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(u[i] << 16);
    out[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u);
  }
}

// Top-k of one token's router over E = 32 * EPL experts, computed by warp 0 and shared through shared memory
// (every thread of the block calls it): the expert of slot k_want and its weight normalized over the K
// winners, picked one by one like a sequential argmax: highest score + bias first, ties to the lower index,
// NaN scores (padded rows) below every number. Lane l holds experts i * 32 + l; a round is a local argmax and
// two redux.sync (max of the order-preserving score bits, then min index among the lanes at that max).
template <int EPL>
__device__ __forceinline__ void route_block(const float* logits, const float* bias, int K, int k_want, int& e_out,
                                            float& w_norm) {
  __shared__ int sel_e[8];
  __shared__ float sel_w[8];
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    float sc[EPL];
    unsigned key[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int e = i * 32 + lane;
      sc[i] = __fdividef(1.f, 1.f + __expf(-logits[e]));
      const float b = sc[i] + bias[e];
      const unsigned u = __float_as_uint(b);
      key[i] = b != b ? 0u : ((u & 0x80000000u) ? ~u : (u | 0x80000000u));
    }
    float tot = 0.f;
    for (int kk = 0; kk < K; ++kk) {
      unsigned bk = key[0];
      int bi = 0;
#pragma unroll
      for (int i = 1; i < EPL; ++i)
        if (key[i] > bk) { bk = key[i]; bi = i; }
      const unsigned mx = __reduce_max_sync(0xffffffffu, bk);
      const unsigned win = __reduce_min_sync(0xffffffffu, bk == mx ? (unsigned)(bi * 32 + lane) : 0xffffffffu);
      float mine = 0.f;
#pragma unroll
      for (int i = 0; i < EPL; ++i)
        if (i * 32 + lane == (int)win) { mine = sc[i]; key[i] = 0u; }
      const float w = __shfl_sync(0xffffffffu, mine, win & 31);
      tot += w;
      if (lane == 0) {
        sel_e[kk] = win;
        sel_w[kk] = w;
      }
    }
    if (lane == 0) sel_w[7] = tot;  // K <= 7 here; slot 7 carries the sum
  }
  __syncthreads();
  e_out = sel_e[k_want];
  w_norm = sel_w[k_want] / sel_w[7];
}

// grid (M * K, N1 / (16 * WARPS)), 32 * WARPS threads. A warp owns 16 rows of W1[e]: 4 groups of 8 lanes, a
// group one row at a time; lane j of a group reads 16-byte chunks j, j + 8, j + 16 of the 384-byte row.
template <int D, int EPL, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) up_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_ld, const float* __restrict__ logits,
    const float* __restrict__ bias, const int8_t* __restrict__ w1, const float* __restrict__ s1, int32_t* ids,
    float* tw, __nv_bfloat16* __restrict__ h, int K, int N1) {
  static_assert(D == 384, "8 lanes x 3 chunks of 16 bytes per row");
  constexpr int E = 32 * EPL;
  const int slot = blockIdx.x, m = slot / K, k = slot % K;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 3, j = lane & 7;
  int e;
  float wk;
  route_block<EPL>(logits + (int64_t)m * E, bias, K, k, e, wk);
  const int row0 = (blockIdx.y * WARPS + warp) * 16;
  const int8_t* wp = w1 + ((int64_t)e * N1 + row0 + g) * D + 16 * j;
  uint4 q[4][3];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < 3; ++c) q[r][c] = ldg128(wp + (int64_t)r * 4 * D + 128 * c);
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    ids[slot] = e;
    tw[slot] = wk;
  }
  float xf[3][16];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    load8_bf16(x + m * x_ld + 128 * c + 16 * j, xf[c]);
    load8_bf16(x + m * x_ld + 128 * c + 16 * j + 8, xf[c] + 8);
  }
  float sc_row[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) sc_row[r] = s1[(int64_t)e * N1 + row0 + 4 * r + g];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float a = dot16(q[r][0], xf[0]) + dot16(q[r][1], xf[1]) + dot16(q[r][2], xf[2]);
    a += __shfl_xor_sync(0xffffffffu, a, 1);
    a += __shfl_xor_sync(0xffffffffu, a, 2);
    a += __shfl_xor_sync(0xffffffffu, a, 4);
    if (j == 0) {
      float v = __bfloat162float(__float2bfloat16(a * sc_row[r]));
      v = fmaxf(v, 0.f);
      h[(int64_t)slot * N1 + row0 + 4 * r + g] = __float2bfloat16(v * v);
    }
  }
}

// grid (M, D2 / RB), 32 * K threads: warp k = top-k slot k, its two half warps 4 rows each of W2[e] (768 bytes:
// lane j of a half reads chunks j, j + 16, j + 32). Every CTA also writes S / gridDim.y shared activations.
template <int N1, int RB>
__global__ void __launch_bounds__(256) down_kernel(
    const __nv_bfloat16* __restrict__ h, const int32_t* __restrict__ ids, const float* __restrict__ tw,
    const int8_t* __restrict__ w2, const float* __restrict__ s2, const __nv_bfloat16* __restrict__ s, int64_t s_ld,
    __nv_bfloat16* __restrict__ cat, float scale, int K, int D2, int S) {
  static_assert(N1 == 768 && RB == 8, "16 lanes x 3 chunks of 16 bytes per row, 4 rows per half warp");
  __shared__ float red[8][RB];
  const int m = blockIdx.x, n0 = blockIdx.y * RB;
  const int k = threadIdx.x >> 5, lane = threadIdx.x & 31, half = lane >> 4, j = lane & 15;
  const int slot = m * K + k;
  const int e = ids[slot];
  const int8_t* wp = w2 + ((int64_t)e * D2 + n0 + half) * N1 + 16 * j;
  uint4 q[4][3];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < 3; ++c) q[r][c] = ldg128(wp + (int64_t)r * 2 * N1 + 256 * c);
  const float wk = tw[slot];
  float hf[3][16];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    load8_bf16(h + (int64_t)slot * N1 + 256 * c + 16 * j, hf[c]);
    load8_bf16(h + (int64_t)slot * N1 + 256 * c + 16 * j + 8, hf[c] + 8);
  }
  float sc_row[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) sc_row[r] = s2[(int64_t)e * D2 + n0 + 2 * r + half];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float a = dot16(q[r][0], hf[0]) + dot16(q[r][1], hf[1]) + dot16(q[r][2], hf[2]);
    a += __shfl_xor_sync(0xffffffffu, a, 1);
    a += __shfl_xor_sync(0xffffffffu, a, 2);
    a += __shfl_xor_sync(0xffffffffu, a, 4);
    a += __shfl_xor_sync(0xffffffffu, a, 8);
    if (j == 0) red[k][2 * r + half] = a * sc_row[r] * wk;
  }
  __syncthreads();
  const int64_t out_ld = D2 + S;
  if (threadIdx.x < RB) {
    float acc = 0.f;
    for (int kk = 0; kk < K; ++kk) acc += red[kk][threadIdx.x];
    cat[m * out_ld + n0 + threadIdx.x] = __float2bfloat16(acc * scale);
  }
  const int per = S / gridDim.y;
  for (int i = threadIdx.x; i < per; i += blockDim.x) {
    const int idx = blockIdx.y * per + i;
    const float v = fmaxf(__bfloat162float(s[m * s_ld + idx]), 0.f);
    cat[m * out_ld + D2 + idx] = __float2bfloat16(v * v);
  }
}

}  // namespace

// x: [M, D] bf16 rows (row stride x_ld), logits [M, E] fp32, bias [E] fp32, w1 [E, N1, D] int8, s1 [E, N1] fp32,
// w2 [E, D2, N1] int8, s2 [E, D2] fp32, s: [M, S] bf16 rows (row stride s_ld). Returns cat [M, D2 + S] bf16.
torch::Tensor moe_small_int8(torch::Tensor x, torch::Tensor logits, torch::Tensor bias, torch::Tensor w1,
                             torch::Tensor s1, torch::Tensor w2, torch::Tensor s2, torch::Tensor s, int64_t top_k,
                             double scale) {
  const int M = x.size(0), D = x.size(1), E = w1.size(0), N1 = w1.size(1), D2 = w2.size(1), S = s.size(1);
  TORCH_CHECK(D == 384 && N1 == 768 && D2 == 384 && E == 256 && w2.size(2) == N1, "Olala latent MoE shapes");
  TORCH_CHECK(top_k >= 1 && top_k <= 7 && M >= 1, "top_k in 1..7");
  TORCH_CHECK(x.stride(1) == 1 && s.stride(1) == 1 && x.stride(0) % 8 == 0 && s.stride(0) % 8 == 0, "row strides");
  TORCH_CHECK(logits.is_contiguous() && logits.size(1) == E && logits.scalar_type() == at::kFloat, "logits");
  TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous() && s1.is_contiguous() && s2.is_contiguous(), "weights");
  TORCH_CHECK(S % (D2 / 8) == 0, "shared width");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && s.scalar_type() == at::kBFloat16, "bf16 activations");
  TORCH_CHECK(w1.scalar_type() == at::kChar && w2.scalar_type() == at::kChar, "int8 experts");
  TORCH_CHECK(s1.scalar_type() == at::kFloat && s2.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat,
              "fp32 scales / bias");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto dev = x.options();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ids = torch::empty({M * top_k}, dev.dtype(at::kInt));
  auto tw = torch::empty({M * top_k}, dev.dtype(at::kFloat));
  auto h = torch::empty({M * top_k, N1}, dev);
  auto cat = torch::empty({M, D2 + S}, dev);
  constexpr int WARPS = 4;
  up_kernel<384, 8, WARPS><<<dim3(M * top_k, N1 / (16 * WARPS)), 32 * WARPS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), logits.data_ptr<float>(),
      bias.data_ptr<float>(), w1.data_ptr<int8_t>(), s1.data_ptr<float>(), ids.data_ptr<int32_t>(),
      tw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), top_k, N1);
  down_kernel<768, 8><<<dim3(M, D2 / 8), 32 * top_k, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()), ids.data_ptr<int32_t>(), tw.data_ptr<float>(),
      w2.data_ptr<int8_t>(), s2.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(s.data_ptr()),
      s.stride(0), reinterpret_cast<__nv_bfloat16*>(cat.data_ptr()), (float)scale, top_k, D2, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return cat;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_small_int8", &moe_small_int8); }
