// Olala routed experts on sm80 / sm86 / sm89: int8 W8A16 per-channel weights, non-gated relu^2, both GEMMs of
// a 32-pair block of moe_align_block_size(block_size=32) in one CTA:
//   H = relu(X @ W1[e]^T * s1[e])^2          (bf16, kept in shared memory)
//   Y = H @ W2[e]^T * s2[e] (* topk weight)  -> out[pair] (bf16); moe_sum adds the top-k rows afterwards.
// Weights stream from global memory straight into registers by whole rows: a warp step loads 16 consecutive
// weight rows x 384 bytes (6 KB contiguous, 16-byte loads), so DRAM sees long bursts, and finishes those two n8
// tiles over that K range. The int8 values become bf16 with the fp32 magic-number conversion; K is permuted
// inside every 64-wide block so a thread's 16 loaded bytes are exactly its B fragments of four k16 MMAs, and X
// and H sit in shared memory with the same permutation (ldmatrix A fragments).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>

namespace {
constexpr int BM = 32;

template <int KIN, int NMID, int WARPS>
struct Cfg {
  static constexpr int THREADS = 32 * WARPS;
  static constexpr int LX = KIN + 8, LH = NMID + 8;  // padded smem rows: ldmatrix rows 16 B apart in bank space
  static constexpr int KB = KIN / 64;                // 64-wide K blocks per step (GEMM 1: all of K, GEMM 2: half)
  static constexpr int S1 = NMID / 16 / WARPS;       // GEMM 1 steps per warp (16 rows of W1 each)
  static constexpr int S2 = 2 * (KIN / 16) / WARPS;  // GEMM 2 steps per warp (16 rows of W2, half of K each)
  static constexpr int SMEM = BM * (LX + LH) * 2;
  static_assert(KIN % 64 == 0 && NMID == 2 * KIN, "a step must cover 384 bytes of a W1 row and of a W2 half row");
  static_assert((NMID / 16) % WARPS == 0 && (KIN / 16) % WARPS == 0, "whole tile pairs per warp");
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg128(const int8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// 4 signed int8 -> two bf16 pairs, exact: fp32 bits 2^23 + (q + 128), minus 2^23 + 128, keep the upper halves
__device__ __forceinline__ void i8x4_bf16(unsigned q, unsigned& lo, unsigned& hi) {
  const unsigned u = q ^ 0x80808080u;
  const float f0 = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650)) - 8388736.f;
  const float f1 = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7651)) - 8388736.f;
  const float f2 = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7652)) - 8388736.f;
  const float f3 = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7653)) - 8388736.f;
  lo = __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632);
  hi = __byte_perm(__float_as_uint(f2), __float_as_uint(f3), 0x7632);
}

// Shared-memory position of element a of a 64-wide K block. Thread c (= lane % 4) loads bytes 16c .. 16c+15 of
// a weight row; bytes 4i .. 4i+3 are its fragment slots {2c, 2c+1, 8+2c, 9+2c} of the block's k16 step i.
__device__ __forceinline__ int kperm(int a) {
  const int c = a >> 4, i = (a >> 2) & 3, off = a & 3;
  return 16 * i + 2 * c + (off & 1) + 8 * (off >> 1);
}

__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<unsigned*>(&h);
}

template <int KIN, int NMID, int WARPS, int MT>
__device__ __forceinline__ void expert_block(__nv_bfloat16* xs, __nv_bfloat16* hs, const int* spid, int nvalid,
                                             int e, const __nv_bfloat16* __restrict__ x,
                                             const int8_t* __restrict__ w1, const float* __restrict__ s1,
                                             const int8_t* __restrict__ w2, const float* __restrict__ s2,
                                             const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ out,
                                             int top_k, int mul_w) {
  using C = Cfg<KIN, NMID, WARPS>;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int arow = (lane & 7) + 8 * ((lane >> 3) & 1), acol = 8 * (lane >> 4);
  const int8_t* w1e = w1 + (long)e * NMID * KIN;
  const int8_t* w2e = w2 + (long)e * KIN * NMID;

  // step s < S1: W1 rows 16 (warp + 8 s) + {g, 8 + g}, all of K. Then W2 rows 16 (warp + 8 (s' / 2)) + {g, 8 + g},
  // K half s' % 2.
  auto load_step = [&](int s, uint4 (&buf)[2][C::KB]) {
    const int8_t* base;
    if (s < C::S1) {
      base = w1e + (long)(16 * (warp + WARPS * s) + g) * KIN + 16 * c;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int kb = 0; kb < C::KB; ++kb) buf[t][kb] = ldg128(base + t * 8 * KIN + 64 * kb);
    } else {
      const int s2i = s - C::S1;
      base = w2e + (long)(16 * (warp + WARPS * (s2i >> 1)) + g) * NMID + (s2i & 1) * KIN + 16 * c;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int kb = 0; kb < C::KB; ++kb) buf[t][kb] = ldg128(base + t * 8 * NMID + 64 * kb);
    }
  };

  uint4 wb[2][2][C::KB];
  load_step(0, wb[0]);

  // X rows of the block's pairs, K-permuted: the 16-byte chunk q of a 64-block -> words 16 (q & 1) + (q >> 1) + 4 i
  for (int i = tid; i < 16 * MT * (KIN / 8); i += C::THREADS) {
    const int r = i / (KIN / 8), ch = i % (KIN / 8);
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < nvalid) v = __ldg(reinterpret_cast<const uint4*>(x + (long)(spid[r] / top_k) * KIN) + ch);
    const int q = ch & 7;
    unsigned* row = reinterpret_cast<unsigned*>(xs + r * C::LX) + 32 * (ch >> 3) + 16 * (q & 1) + (q >> 1);
    row[0] = v.x;
    row[4] = v.y;
    row[8] = v.z;
    row[12] = v.w;
  }
  __syncthreads();

  float acc[2][MT][4];
#pragma unroll
  for (int s = 0; s < C::S1 + C::S2; ++s) {
    if (s + 1 < C::S1 + C::S2) load_step(s + 1, wb[(s + 1) & 1]);
    const bool g1 = s < C::S1;
    const int half = g1 ? 0 : (s - C::S1) & 1;
    if (g1 || half == 0) {
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[t][mi][q] = 0.f;
    }
    if (s == C::S1) __syncthreads();  // H complete
    const __nv_bfloat16* as = g1 ? xs : hs;
    const int lda = g1 ? C::LX : C::LH, k0 = half * KIN;
#pragma unroll
    for (int kk = 0; kk < 4 * C::KB; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) ldsm_x4(a[mi], as + (16 * mi + arow) * lda + k0 + 16 * kk + acol);
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        unsigned b0, b1;
        i8x4_bf16(word(wb[s & 1][t][kk >> 2], kk & 3), b0, b1);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) mma(acc[t][mi], a[mi], b0, b1);
      }
    }
    if (g1) {  // relu^2 tile -> H
      const float* s1e = s1 + (long)e * NMID;
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int h = 16 * (warp + WARPS * s) + 8 * t + 2 * c;
        const float2 sc = __ldg(reinterpret_cast<const float2*>(s1e + h));
        const int ph = (h & ~63) + kperm(h & 63);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          const float v0 = fmaxf(acc[t][mi][0] * sc.x, 0.f), v1 = fmaxf(acc[t][mi][1] * sc.y, 0.f);
          const float v2 = fmaxf(acc[t][mi][2] * sc.x, 0.f), v3 = fmaxf(acc[t][mi][3] * sc.y, 0.f);
          const int r = 16 * mi + g;
          *reinterpret_cast<unsigned*>(hs + r * C::LH + ph) = pack_bf16(v0 * v0, v1 * v1);
          *reinterpret_cast<unsigned*>(hs + (r + 8) * C::LH + ph) = pack_bf16(v2 * v2, v3 * v3);
        }
      }
    } else if (half == 1) {  // output tile
      const float* s2e = s2 + (long)e * KIN;
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int n = 16 * (warp + WARPS * ((s - C::S1) >> 1)) + 8 * t + 2 * c;
        const float2 sc = __ldg(reinterpret_cast<const float2*>(s2e + n));
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int r = 16 * mi + g + 8 * hh;
            if (r < nvalid) {
              const int p = spid[r];
              const float w = mul_w ? __ldg(topk_w + p) : 1.f;
              *reinterpret_cast<unsigned*>(out + (long)p * KIN + n) =
                  pack_bf16(acc[t][mi][2 * hh] * sc.x * w, acc[t][mi][2 * hh + 1] * sc.y * w);
            }
          }
      }
    }
  }
}

template <int KIN, int NMID, int WARPS>
__global__ void __launch_bounds__(32 * WARPS, 1)
    moe_relu2_int8_kernel(const __nv_bfloat16* __restrict__ x, const int8_t* __restrict__ w1,
                          const float* __restrict__ s1, const int8_t* __restrict__ w2, const float* __restrict__ s2,
                          const float* __restrict__ topk_w, const int* __restrict__ sorted_ids,
                          const int* __restrict__ expert_ids, const int* __restrict__ ntpp,
                          __nv_bfloat16* __restrict__ out, int num_pairs, int top_k, int mul_w) {
  using C = Cfg<KIN, NMID, WARPS>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ int spid[BM];
  __shared__ int s_nvalid;
  const int blk = blockIdx.x;
  if (blk * BM >= __ldg(ntpp)) return;
  const int e = __ldg(expert_ids + blk);
  if (threadIdx.x < 32) {  // padding entries (= num_pairs) close every expert segment: valid rows are a prefix
    const int p = __ldg(sorted_ids + blk * BM + threadIdx.x);
    spid[threadIdx.x] = p;
    const unsigned m = __ballot_sync(0xffffffffu, p < num_pairs);
    if (threadIdx.x == 0) s_nvalid = __popc(m);
  }
  __syncthreads();
  const int nvalid = s_nvalid;
  if (e < 0) {  // expert not on this rank
    for (int i = threadIdx.x; i < nvalid * KIN / 2; i += C::THREADS)
      reinterpret_cast<unsigned*>(out + (long)spid[i / (KIN / 2)] * KIN)[i % (KIN / 2)] = 0u;
    return;
  }
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* hs = xs + BM * C::LX;
  if (nvalid > 16)
    expert_block<KIN, NMID, WARPS, 2>(xs, hs, spid, nvalid, e, x, w1, s1, w2, s2, topk_w, out, top_k, mul_w);
  else
    expert_block<KIN, NMID, WARPS, 1>(xs, hs, spid, nvalid, e, x, w1, s1, w2, s2, topk_w, out, top_k, mul_w);
}
}  // namespace

template <int KIN, int NMID, int WARPS, typename Args>
void launch(int grid, const Args& args) {
  using C = Cfg<KIN, NMID, WARPS>;
  auto kern = moe_relu2_int8_kernel<KIN, NMID, WARPS>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess);
    attr = true;
  }
  std::apply([&](auto... a) { kern<<<grid, C::THREADS, C::SMEM, c10::cuda::getCurrentCUDAStream()>>>(a...); }, args);
}

void moe_relu2_int8(torch::Tensor x, torch::Tensor w1, torch::Tensor s1, torch::Tensor w2, torch::Tensor s2,
                    torch::Tensor topk_w, torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor ntpp,
                    torch::Tensor out, int64_t top_k, bool mul_w) {
  constexpr int KIN = 384, NMID = 768;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(-1) == KIN, "x: bf16 [T, 384]");
  TORCH_CHECK(w1.scalar_type() == at::kChar && w1.is_contiguous() && w1.size(1) == NMID && w1.size(2) == KIN);
  TORCH_CHECK(w2.scalar_type() == at::kChar && w2.is_contiguous() && w2.size(1) == KIN && w2.size(2) == NMID);
  TORCH_CHECK(s1.scalar_type() == at::kFloat && s1.is_contiguous() && s1.numel() == w1.size(0) * NMID);
  TORCH_CHECK(s2.scalar_type() == at::kFloat && s2.is_contiguous() && s2.numel() == w2.size(0) * KIN);
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.is_contiguous());
  TORCH_CHECK(sorted_ids.scalar_type() == at::kInt && expert_ids.scalar_type() == at::kInt);
  // blocks past num_tokens_post_padded (a multiple of 32) exit before reading sorted_ids
  TORCH_CHECK(sorted_ids.numel() > (expert_ids.numel() - 1) * BM, "moe_align_block_size must use block_size 32");
  const long num_pairs = x.size(0) * top_k;
  TORCH_CHECK(topk_w.numel() == num_pairs && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
              out.numel() == num_pairs * KIN);
  const int grid = (int)expert_ids.numel();
  if (grid == 0 || num_pairs == 0) return;
  const auto args = std::make_tuple(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w1.data_ptr<int8_t>(), s1.data_ptr<float>(),
      w2.data_ptr<int8_t>(), s2.data_ptr<float>(), topk_w.data_ptr<float>(), sorted_ids.data_ptr<int>(),
      expert_ids.data_ptr<int>(), ntpp.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      (int)num_pairs, (int)top_k, mul_w ? 1 : 0);
  // 8 warps for decode-sized batches, 12 (more loads in flight per SM) from 512 tokens on (measured on A4500)
  if (num_pairs >= 512 * top_k)
    launch<KIN, NMID, 12>(grid, args);
  else
    launch<KIN, NMID, 8>(grid, args);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_relu2_int8", &moe_relu2_int8); }
