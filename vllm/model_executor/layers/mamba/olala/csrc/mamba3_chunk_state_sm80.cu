// Olala Mamba-3 MIMO prefill: chunk-local states on sm80 / sm86 / sm89 (mma.sync), the Ampere / Ada
// counterpart of the sm90 chunk_state in mamba3_scan_prefill.cu (same arguments, same math):
//   Sloc[c, h][n][p] = sum_j khat_j[n] * PsiV_j[p] * ts_j * exp(dtot[c, h] - dacs_j)     (fp32, (NC, H, N, P))
// over the rows j = token * R + rank of chunk c, with khat = (K + kbias) rotated on the (n, n + N/2), n < N/4
// pairs by the chunk-local phase thl. One CTA per (chunk, head), 16-token sub-blocks (64 rows): the global
// loads of the next sub-block are issued before the MMAs of the current one; every thread prepares 2 ranks x
// 16 khat columns of one token (bias in bf16, rotation in fp32, as the Triton reference) and 2 ranks x 8
// PsiV * w columns, 8 warps accumulate the 128 x 64 state (32 x 32 each) with m16n8k16 bf16 MMAs.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32;
constexpr int CC = 16, ROWS = CC * R;  // 64 rows per sub-block
constexpr int THREADS = 256;
// padded smem rows (bf16): the 8 rows an ldmatrix reads land 16 B apart in the bank space
constexpr int LDA = N + 8, LDB = P + 8, LKB = N + 8;

struct Smem {
  __nv_bfloat16 a[ROWS * LDA];  // khat [row][n] (A^T)
  __nv_bfloat16 b[ROWS * LDB];  // PsiV * w [row][p]
  __nv_bfloat16 kb[R * LKB];
  float psi[R * P];
};

union U8 {
  uint4 u;
  __nv_bfloat162 h2[4];
};

struct Raw {        // one thread's global inputs of a sub-block
  uint4 k[2][2];    // ranks 2 rh + {0, 1}, columns 8 j and 64 + 8 j
  float4 th[2];     // thl of columns 8 j .. 8 j + 7 (rotated groups, j < 4)
  uint4 v;          // V[token][8 pc .. 8 pc + 7]
  float ts, dacs;
};

// 8 bf16 of a strided row: one 16-byte load when aligned, else the two aligned 16-byte pieces around it and a
// funnel shift (the in_proj output interleaves heads with an odd stride; the alignment is uniform per CTA)
__device__ __forceinline__ uint4 ld8(const __nv_bfloat16* __restrict__ p, bool vec) {
  if (vec) return __ldg(reinterpret_cast<const uint4*>(p));
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const unsigned off = (unsigned)(a & 15);  // even: 0 .. 14 bytes
  const uint4* base = reinterpret_cast<const uint4*>(a - off);
  const uint4 u0 = __ldg(base);
  if (off == 0) return u0;
  const uint4 u1 = __ldg(base + 1);
  const unsigned w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
  const unsigned wi = off >> 2, sh = (off & 3) * 8;
  unsigned o[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const unsigned lo = wi == 0 ? w[e] : wi == 1 ? w[e + 1] : wi == 2 ? w[e + 2] : w[e + 3];
    const unsigned hi = wi == 0 ? w[e + 1] : wi == 1 ? w[e + 2] : wi == 2 ? w[e + 3] : w[e + 4];
    o[e] = __funnelshift_r(lo, hi, sh);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_bf16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void load_raw(Raw& rw, int tid, int t0, int nv, int h, int T, int H, long ldv, long lhv,
                                         bool vec,
                                         const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
                                         const float* __restrict__ thl, const float* __restrict__ dacs,
                                         const float* __restrict__ ts) {
  const int tok = tid >> 4, j = tid & 7, rh = (tid >> 3) & 1, pc = (tid >> 1) & 7;
  const bool ok = tok < nv;
  const long t = t0 + tok;
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const __nv_bfloat16* kr = K + (t * R + 2 * rh + rr) * N + 8 * j;
    rw.k[rr][0] = ok ? __ldg(reinterpret_cast<const uint4*>(kr)) : z;
    rw.k[rr][1] = ok ? __ldg(reinterpret_cast<const uint4*>(kr + 64)) : z;
  }
  if (j < 4) {
    const float4* tp = reinterpret_cast<const float4*>(thl + (t * H + h) * NQ + 8 * j);
    const float4 zf = make_float4(0.f, 0.f, 0.f, 0.f);
    rw.th[0] = ok ? __ldg(tp) : zf;
    rw.th[1] = ok ? __ldg(tp + 1) : zf;
  }
  rw.v = ok ? ld8(V + t * ldv + h * lhv + 8 * pc, vec) : z;
  rw.ts = ok ? __ldg(ts + (long)h * T + t) : 0.f;
  rw.dacs = ok ? __ldg(dacs + (long)h * T + t) : 0.f;
}

__device__ __forceinline__ void store_operands(Smem& S, const Raw& rw, int tid, float dl) {
  const int tok = tid >> 4, j = tid & 7, rh = (tid >> 3) & 1;
  const bool rot = j < 4;
  float cs[8], sn[8];
  if (rot) {
    const float th[8] = {rw.th[0].x, rw.th[0].y, rw.th[0].z, rw.th[0].w,
                         rw.th[1].x, rw.th[1].y, rw.th[1].z, rw.th[1].w};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float a = th[e] - 6.283185307179586f * floorf(th[e] * 0.15915494309189535f);
      __sincosf(a, &sn[e], &cs[e]);
    }
  }
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int r = 2 * rh + rr, row = tok * R + r;
    U8 k0, k1, b0, b1;
    k0.u = rw.k[rr][0];
    k1.u = rw.k[rr][1];
    b0.u = *reinterpret_cast<const uint4*>(&S.kb[r * LKB + 8 * j]);
    b1.u = *reinterpret_cast<const uint4*>(&S.kb[r * LKB + 64 + 8 * j]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      k0.h2[e] = __hadd2(k0.h2[e], b0.h2[e]);
      k1.h2[e] = __hadd2(k1.h2[e], b1.h2[e]);
    }
    if (rot) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 y0 = __bfloat1622float2(k0.h2[e]), y1 = __bfloat1622float2(k1.h2[e]);
        const float ca = cs[2 * e], cb = cs[2 * e + 1], sa = sn[2 * e], sb = sn[2 * e + 1];
        k0.h2[e] = __floats2bfloat162_rn(ca * y0.x - sa * y1.x, cb * y0.y - sb * y1.y);
        k1.h2[e] = __floats2bfloat162_rn(sa * y0.x + ca * y1.x, sb * y0.y + cb * y1.y);
      }
    }
    *reinterpret_cast<uint4*>(&S.a[row * LDA + 8 * j]) = k0.u;
    *reinterpret_cast<uint4*>(&S.a[row * LDA + 64 + 8 * j]) = k1.u;
  }
  const int pc = (tid >> 1) & 7, vrh = tid & 1;
  const float w = rw.ts * __expf(dl - rw.dacs);  // ts = 0 past the chunk end
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int r = 2 * vrh + rr, row = tok * R + r;
    const float4* pp = reinterpret_cast<const float4*>(&S.psi[r * P + 8 * pc]);
    const float4 pa = pp[0], pb = pp[1];
    const float ps[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
    U8 v;
    v.u = rw.v;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 x = __bfloat1622float2(v.h2[e]);
      v.h2[e] = __floats2bfloat162_rn(x.x * ps[2 * e] * w, x.y * ps[2 * e + 1] * w);
    }
    *reinterpret_cast<uint4*>(&S.b[row * LDB + 8 * pc]) = v.u;
  }
}

__global__ void __launch_bounds__(THREADS, 2) chunk_state_sm80_kernel(
    const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, const float* __restrict__ kbias,
    const float* __restrict__ psi, const float* __restrict__ thl, const float* __restrict__ dacs,
    const float* __restrict__ ts, const float* __restrict__ dtot, const int* __restrict__ ch_start,
    const int* __restrict__ ch_len, float* __restrict__ sloc, int T, int H, long ldv, long lhv) {
  __shared__ __align__(16) Smem S;
  const int tid = threadIdx.x, item = blockIdx.x, c = item / H, h = item % H;
  const int cs = ch_start[c], cl = ch_len[c], nsb = (cl + CC - 1) / CC;
  const float dl = dtot[item];
  const bool vec = ((ldv | lhv) & 7) == 0;
  for (int i = tid; i < R * N; i += THREADS) S.kb[(i / N) * LKB + i % N] = __float2bfloat16(kbias[(long)h * R * N + i]);
  for (int i = tid; i < R * P; i += THREADS) S.psi[i] = psi[(long)h * R * P + i];
  Raw rw;
  load_raw(rw, tid, cs, min(CC, cl), h, T, H, ldv, lhv, vec, K, V, thl, dacs, ts);

  const int lane = tid & 31, warp = tid >> 5, wm = warp & 3, wn = warp >> 2, q = lane >> 3;
  float acc[2][4][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;

  for (int it = 0; it < nsb; ++it) {
    __syncthreads();
    store_operands(S, rw, tid, dl);
    __syncthreads();
    if (it + 1 < nsb)
      load_raw(rw, tid, cs + (it + 1) * CC, min(CC, cl - (it + 1) * CC), h, T, H, ldv, lhv, vec, K, V, thl, dacs, ts);
#pragma unroll
    for (int ks = 0; ks < ROWS / 16; ++ks) {
      unsigned a[2][4], b[4][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int k = 16 * ks + 8 * (q >> 1) + (lane & 7), m = 32 * wm + 16 * mt + 8 * (q & 1);
        ldsm_x4_t(a[mt][0], a[mt][1], a[mt][2], a[mt][3], &S.a[k * LDA + m]);
      }
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        const int k = 16 * ks + 8 * (q & 1) + (lane & 7), n = 32 * wn + 16 * np + 8 * (q >> 1);
        ldsm_x4_t(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1], &S.b[k * LDB + n]);
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_bf16(acc[mt][nt], a[mt], b[nt][0], b[nt][1]);
    }
  }
  float* dst = sloc + (long)item * N * P;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int m = 32 * wm + 16 * mt + (lane >> 2), p = 32 * wn + 8 * nt + 2 * (lane & 3);
      *reinterpret_cast<float2*>(&dst[m * P + p]) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<float2*>(&dst[(m + 8) * P + p]) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
}
}  // namespace

void mamba3_chunk_state_sm80(torch::Tensor K, torch::Tensor V, torch::Tensor kbias, torch::Tensor psi,
                             torch::Tensor thl, torch::Tensor dacs, torch::Tensor ts, torch::Tensor dtot,
                             torch::Tensor ch_start, torch::Tensor ch_len, torch::Tensor sloc, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && K.size(-1) == N && K.size(-2) == R && thl.size(-1) == NQ);
  TORCH_CHECK(K.scalar_type() == at::kBFloat16 && V.scalar_type() == at::kBFloat16);
  TORCH_CHECK(K.is_contiguous() && V.stride(2) == 1 && thl.is_contiguous() && sloc.is_contiguous());
  TORCH_CHECK(kbias.is_contiguous() && psi.is_contiguous() && dacs.is_contiguous() && ts.is_contiguous());
  if (NC * H == 0) return;
  chunk_state_sm80_kernel<<<(int)(NC * H), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
      kbias.data_ptr<float>(), psi.data_ptr<float>(), thl.data_ptr<float>(), dacs.data_ptr<float>(),
      ts.data_ptr<float>(), dtot.data_ptr<float>(), ch_start.data_ptr<int>(), ch_len.data_ptr<int>(),
      sloc.data_ptr<float>(), T, H, V.stride(0), V.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("chunk_state", &mamba3_chunk_state_sm80); }
