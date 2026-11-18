// Olala Mamba-3 MIMO prefill, output pass on sm80 / sm86 / sm89 (mma.sync): the Ampere / Ada counterpart of the
// sm90 `scan` in mamba3_scan_prefill.cu (same arguments and math, same bf16 roundings), with 8-token sub-blocks
// (32 MIMO rows, row = token * R + rank; da_cs / da_cs_rev are 8-token group cumsums). Every 128-token chunk
// (one head) starts from its entering state S_in (fp32, [P][N], from the Triton state pass):
//   o   = exp(da_cs) * (qhat @ S)                                     inter-chunk
//   G   = qhat @ (khat * ts * e)^T, masked: earlier token j -> exp(da_cs_i - da_cs_j) / e_j, same token -> gamma/ts / e_j
//   o  += G @ PsiV ;  out[t] = sum_r (o[t, r] + D PsiV[t, r]) * phi_r * silu(z_t * zeta_r)
//   S   = exp(da_cs_last) * S + (khat * ts * e)^T @ PsiV                 e_j = exp(max(da_cs_rev_j, -80))
// One CTA (8 warps, ~47 KB of smem: two CTAs per SM overlap each other's phases) per (chunk, head). The fp32
// state lives in registers (warp w: n rows 32 (w & 3), p cols 32 (w >> 2)), its bf16 copy in smem for qhat @ S;
// o is 16 x 16 and G 16 x 8 per warp. The operands of a sub-block are prepared from registers loaded one
// sub-block ahead (bias in bf16, rotation in fp32, PsiV / gate in bf16), one rank per thread.
// V and Z are read through row / head strides, so they can be the (head-interleaved) in_proj output itself.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32, C = 8, ROWS = C * R;
constexpr int THREADS = 256;
constexpr int LDG = ROWS + 8;  // padded G rows: the 8 rows an ldmatrix reads land 16 B apart in the bank space
constexpr float LOG_EMIN = 80.f;

struct Smem {
  __nv_bfloat16 q[ROWS * N];     // qhat [row][n]
  __nv_bfloat16 k[ROWS * N];     // khat * ts * e [row][n]
  __nv_bfloat16 sb[N * P];       // state, bf16 [n][p]
  __nv_bfloat16 pv[ROWS * P];    // PsiV [row][p]
  __nv_bfloat16 gz[ROWS * P];    // phi_r * silu(z * zeta_r) [row][p]
  __nv_bfloat16 g[ROWS * LDG];   // masked G [row i][row j]
  float tok[C][4];               // da_cs, da_cs_rev, ts, gamma per token
  __nv_bfloat16 qb[R * N], kb[R * N], psi[R * P];
  float phi[R * P], zeta[R * P];
};
// 16 B chunks XOR-swizzled by row: the 8 rows an ldmatrix reads hit 8 different bank groups
__device__ __forceinline__ int swn(int row, int col) { return row * N + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }
__device__ __forceinline__ int swp(int row, int col) { return row * P + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }

union U8 {
  uint4 u;
  __nv_bfloat162 h2[4];
};

struct Raw {
  uint4 q[2], k[2];  // rank (tid >> 3) & 3 of token tid >> 5, columns 8 j and 64 + 8 j
  float4 th[2];      // thl, columns 8 j .. 8 j + 7 (j < 4)
  uint4 v, z;        // columns 8 j .. 8 j + 7
  float dcs, dcsr, ts, gam;
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
__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// thread: token tid >> 5, rank (tid >> 3) & 3, column chunk j = tid & 7 (qhat / khat: 8 j and 64 + 8 j;
// PsiV / gate: 8 j)
__device__ __forceinline__ void load_raw(Raw& rw, int tid, int t0, int nv, int h, int T, int H, long ldv, long ldz,
                                         long lhv, long lhz, bool vec,
                                         const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                         const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ Z,
                                         const float* __restrict__ thl, const float* __restrict__ dcs8,
                                         const float* __restrict__ dcsr8, const float* __restrict__ ts_g,
                                         const float* __restrict__ gam_g) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7;
  const bool ok = tok < nv;
  const long t = t0 + tok;
  const uint4 z0 = make_uint4(0u, 0u, 0u, 0u);
  const long off = (t * R + r) * N + 8 * j;
  rw.q[0] = ok ? __ldg(reinterpret_cast<const uint4*>(Q + off)) : z0;
  rw.q[1] = ok ? __ldg(reinterpret_cast<const uint4*>(Q + off + 64)) : z0;
  rw.k[0] = ok ? __ldg(reinterpret_cast<const uint4*>(K + off)) : z0;
  rw.k[1] = ok ? __ldg(reinterpret_cast<const uint4*>(K + off + 64)) : z0;
  if (j < 4) {
    const float4* tp = reinterpret_cast<const float4*>(thl + (t * H + h) * NQ + 8 * j);
    const float4 zf = make_float4(0.f, 0.f, 0.f, 0.f);
    rw.th[0] = ok ? __ldg(tp) : zf;
    rw.th[1] = ok ? __ldg(tp + 1) : zf;
  }
  rw.v = ok ? ld8(V + t * ldv + h * lhv + 8 * j, vec) : z0;
  rw.z = ok ? ld8(Z + t * ldz + h * lhz + 8 * j, vec) : z0;
  const long ht = (long)h * T + t;
  rw.dcs = ok ? __ldg(dcs8 + ht) : 0.f;
  rw.dcsr = ok ? __ldg(dcsr8 + ht) : 0.f;
  rw.ts = ok ? __ldg(ts_g + ht) : 0.f;
  rw.gam = ok ? __ldg(gam_g + ht) : 0.f;
}

__device__ __forceinline__ void store_operands(Smem& S, const Raw& rw, int tid, const float* a0, bool last_tok,
                                               __nv_bfloat16* __restrict__ fk, long fk_rank_stride) {
  const int tok = tid >> 5, r = (tid >> 3) & 3, j = tid & 7, row = tok * R + r;
  {
    U8 q0, q1, k0, k1, bq0, bq1, bk0, bk1;
    q0.u = rw.q[0];
    q1.u = rw.q[1];
    k0.u = rw.k[0];
    k1.u = rw.k[1];
    bq0.u = *reinterpret_cast<const uint4*>(&S.qb[r * N + 8 * j]);
    bq1.u = *reinterpret_cast<const uint4*>(&S.qb[r * N + 64 + 8 * j]);
    bk0.u = *reinterpret_cast<const uint4*>(&S.kb[r * N + 8 * j]);
    bk1.u = *reinterpret_cast<const uint4*>(&S.kb[r * N + 64 + 8 * j]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      q0.h2[e] = __hadd2(q0.h2[e], bq0.h2[e]);
      q1.h2[e] = __hadd2(q1.h2[e], bq1.h2[e]);
      k0.h2[e] = __hadd2(k0.h2[e], bk0.h2[e]);
      k1.h2[e] = __hadd2(k1.h2[e], bk1.h2[e]);
    }
    if (j < 4) {
      const float th[8] = {rw.th[0].x, rw.th[0].y, rw.th[0].z, rw.th[0].w,
                           rw.th[1].x, rw.th[1].y, rw.th[1].z, rw.th[1].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float ca, sa, cb, sb;
        float a = th[2 * e] + a0[2 * e], b = th[2 * e + 1] + a0[2 * e + 1];
        a = a - 6.283185307179586f * floorf(a * 0.15915494309189535f);
        b = b - 6.283185307179586f * floorf(b * 0.15915494309189535f);
        __sincosf(a, &sa, &ca);
        __sincosf(b, &sb, &cb);
        const float2 x0 = __bfloat1622float2(q0.h2[e]), x1 = __bfloat1622float2(q1.h2[e]);
        const float2 y0 = __bfloat1622float2(k0.h2[e]), y1 = __bfloat1622float2(k1.h2[e]);
        q0.h2[e] = __floats2bfloat162_rn(ca * x0.x - sa * x1.x, cb * x0.y - sb * x1.y);
        q1.h2[e] = __floats2bfloat162_rn(sa * x0.x + ca * x1.x, sb * x0.y + cb * x1.y);
        k0.h2[e] = __floats2bfloat162_rn(ca * y0.x - sa * y1.x, cb * y0.y - sb * y1.y);
        k1.h2[e] = __floats2bfloat162_rn(sa * y0.x + ca * y1.x, sb * y0.y + cb * y1.y);
      }
    }
    if (last_tok) {
      *reinterpret_cast<uint4*>(fk + r * fk_rank_stride + 8 * j) = k0.u;
      *reinterpret_cast<uint4*>(fk + r * fk_rank_stride + 64 + 8 * j) = k1.u;
    }
    // khat * (ts * e_j) in one bf16 product (ts itself is held in bf16)
    const __nv_bfloat162 tse2 = __float2bfloat162_rn(rbf(rw.ts) * __expf(fmaxf(rw.dcsr, -LOG_EMIN)));
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      k0.h2[e] = __hmul2(k0.h2[e], tse2);
      k1.h2[e] = __hmul2(k1.h2[e], tse2);
    }
    *reinterpret_cast<uint4*>(&S.q[swn(row, 8 * j)]) = q0.u;
    *reinterpret_cast<uint4*>(&S.q[swn(row, 64 + 8 * j)]) = q1.u;
    *reinterpret_cast<uint4*>(&S.k[swn(row, 8 * j)]) = k0.u;
    *reinterpret_cast<uint4*>(&S.k[swn(row, 64 + 8 * j)]) = k1.u;
  }
  {
    const int col = 8 * j;
    U8 va, za, ps, g8;
    va.u = rw.v;
    za.u = rw.z;
    ps.u = *reinterpret_cast<const uint4*>(&S.psi[r * P + col]);
    const float4* zp = reinterpret_cast<const float4*>(&S.zeta[r * P + col]);
    const float4* fp = reinterpret_cast<const float4*>(&S.phi[r * P + col]);
    const float4 za4 = zp[0], zb4 = zp[1], fa4 = fp[0], fb4 = fp[1];
    const float zt[8] = {za4.x, za4.y, za4.z, za4.w, zb4.x, zb4.y, zb4.z, zb4.w};
    const float ft[8] = {fa4.x, fa4.y, fa4.z, fa4.w, fb4.x, fb4.y, fb4.z, fb4.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      va.h2[e] = __hmul2(va.h2[e], ps.h2[e]);
      const float2 z = __bfloat1622float2(za.h2[e]);
      const float g0 = 0.5f * z.x * zt[2 * e], g1 = 0.5f * z.y * zt[2 * e + 1];
      const float s0 = rbf(g0 * tanh_approx(g0) + g0), s1 = rbf(g1 * tanh_approx(g1) + g1);  // silu in bf16
      g8.h2[e] = __floats2bfloat162_rn(ft[2 * e] * s0, ft[2 * e + 1] * s1);
    }
    *reinterpret_cast<uint4*>(&S.pv[swp(row, col)]) = va.u;
    *reinterpret_cast<uint4*>(&S.gz[swp(row, col)]) = g8.u;
  }
  if ((tid & 31) == 0) {
    S.tok[tok][0] = rw.dcs;
    S.tok[tok][1] = rw.dcsr;
    S.tok[tok][2] = rw.ts;
    S.tok[tok][3] = rw.gam;
  }
}

__global__ void __launch_bounds__(THREADS, 2) scan_sm80_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,  // (T, R, N)
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ Z,  // (T, H, P), row strides ldv / ldz
    const float* __restrict__ qbias, const float* __restrict__ kbias,          // (H, R, N)
    const float* __restrict__ psi, const float* __restrict__ phi, const float* __restrict__ zeta,  // (H, R, P)
    const float* __restrict__ Dskip,                                           // (H,)
    const float* __restrict__ thl, const float* __restrict__ ain,              // (T, H, NQ), (NC, H, NQ)
    const float* __restrict__ dcs8, const float* __restrict__ dcsr8,           // (H, T), 8-token groups
    const float* __restrict__ ts_g, const float* __restrict__ gam_g,           // (H, T)
    const int* __restrict__ ch_start, const int* __restrict__ ch_len, const int* __restrict__ ch_last,
    const float* __restrict__ sin_t,                                           // (NC, H, P, N)
    __nv_bfloat16* __restrict__ out,                                           // (T, H, P)
    __nv_bfloat16* __restrict__ fin_k,                                         // (NC, R, H, N)
    int T, int H, long ldv, long ldz, long lhv, long lhz) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, item = blockIdx.x, c = item / H, h = item % H;
  const int cs = ch_start[c], cl = ch_len[c], nsb = (cl + C - 1) / C, last_seq = ch_last[c];
  const float dsk = Dskip[h];
  const bool vec = ((ldv | ldz | lhv | lhz) & 7) == 0;
  for (int i = tid; i < R * N; i += THREADS) {
    S.qb[i] = __float2bfloat16(qbias[(long)h * R * N + i]);
    S.kb[i] = __float2bfloat16(kbias[(long)h * R * N + i]);
  }
  for (int i = tid; i < R * P; i += THREADS) {
    const long o = (long)h * R * P + i;
    S.psi[i] = __float2bfloat16(psi[o]);
    S.phi[i] = rbf(phi[o]);
    S.zeta[i] = zeta[o];
  }
  float a0[8];
  {
    const int j = tid & 7;
#pragma unroll
    for (int e = 0; e < 8; ++e) a0[e] = j < 4 ? ain[(long)item * NQ + 8 * j + e] : 0.f;
  }
  const int lane = tid & 31, warp = tid >> 5, g_ = lane >> 2, cq = lane & 3, q4 = lane >> 3;
  const int sr = warp & 3, sc = warp >> 2;  // state tile: n rows 32 sr, p cols 32 sc
  const int orow = warp & 1, ocol = warp >> 1;  // o: rows 16 orow, p cols 16 ocol; G: rows 16 orow, j cols 8 ocol
  // fp32 state: rows n = 32 sr + 16 mt + g_ (+ 8), cols p = 32 sc + 8 nt + 2 cq (+ 1)
  float st[2][4][4];
  {
    const float* sbg = sin_t + (long)item * P * N;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int n = 32 * sr + 16 * mt + g_ + 8 * (e >> 1), p = 32 * sc + 8 * nt + 2 * cq + (e & 1);
          st[mt][nt][e] = __ldg(sbg + (long)p * N + n);
        }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          const int n = 32 * sr + 16 * mt + g_ + 8 * eh, p = 32 * sc + 8 * nt + 2 * cq;
          *reinterpret_cast<unsigned*>(&S.sb[swp(n, p)]) = pack_bf16(st[mt][nt][2 * eh], st[mt][nt][2 * eh + 1]);
        }
  }
  Raw rw;
  load_raw(rw, tid, cs, min(C, cl), h, T, H, ldv, ldz, lhv, lhz, vec, Q, K, V, Z, thl, dcs8, dcsr8, ts_g, gam_g);
  const long fk_rank_stride = (long)H * N;
  __nv_bfloat16* fk = fin_k + ((long)(c * R) * H + h) * N;
  const int ri = 16 * orow + g_, ti_lo = ri >> 2, ti_hi = ti_lo + 2;

  for (int it = 0; it < nsb; ++it) {
    const int t0 = cs + it * C, nv = min(C, cl - it * C);
    __syncthreads();  // previous sub-block done with q / k / pv / gz / g / tok; S.sb written
    store_operands(S, rw, tid, a0, last_seq >= 0 && it + 1 == nsb && (tid >> 5) == nv - 1, fk, fk_rank_stride);
    __syncthreads();
    if (it + 1 < nsb)
      load_raw(rw, tid, t0 + C, min(C, cl - (it + 1) * C), h, T, H, ldv, ldz, lhv, lhz, vec, Q, K, V, Z, thl, dcs8,
               dcsr8, ts_g, gam_g);

    // ---- phase 1: o = qhat @ S (rows 16 orow, p cols 16 ocol), G = qhat @ k'^T (rows 16 orow, j cols 8 ocol)
    float o[2][4], ga[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) o[0][e] = o[1][e] = ga[e] = 0.f;
#pragma unroll
    for (int kp = 0; kp < N / 32; ++kp) {  // two k-steps of 16 per iteration (one x4 of k' covers both)
      unsigned kb4[4];
      ldsm_x4(kb4, &S.k[swn(8 * ocol + (lane & 7), 32 * kp + 8 * (lane >> 3))]);
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        const int ks = 2 * kp + kh;
        unsigned a[4], b[4];
        ldsm_x4(a, &S.q[swn(16 * orow + (lane & 15), 16 * ks + 8 * (lane >> 4))]);
        ldsm_x4_t(b, &S.sb[swp(16 * ks + 8 * (q4 & 1) + (lane & 7), 16 * ocol + 8 * (q4 >> 1))]);
        mma(o[0], a, b[0], b[1]);
        mma(o[1], a, b[2], b[3]);
        mma(ga, a, kb4[2 * kh], kb4[2 * kh + 1]);
      }
    }
    {
      const float elo = __expf(S.tok[ti_lo][0]), ehi = __expf(S.tok[ti_hi][0]);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        o[nt][0] *= elo;
        o[nt][1] *= elo;
        o[nt][2] *= ehi;
        o[nt][3] *= ehi;
      }
      const int j0 = 8 * ocol + 2 * cq, tj = j0 >> 2;
      const float dj = S.tok[tj][0], lej = fmaxf(S.tok[tj][1], -LOG_EMIN);
#pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int ti = eh ? ti_hi : ti_lo, row = ri + 8 * eh;
        const float tsi = ti < nv ? rbf(S.tok[ti][2]) : 0.f;  // the trap scale is held in bf16
        const float dg = (ti < nv && tsi > 0.f) ? S.tok[ti][3] / tsi : 0.f;
        const float cf = ti > tj ? __expf(fminf(S.tok[ti][0] - dj, 0.f) - lej) : (ti == tj ? dg * __expf(-lej) : 0.f);
        *reinterpret_cast<unsigned*>(&S.g[row * LDG + j0]) = pack_bf16(ga[2 * eh] * cf, ga[2 * eh + 1] * cf);
      }
    }
    __syncthreads();  // G complete; every warp done reading S.sb

    // ---- phase 2: o += G @ PsiV; S = dec S + k'^T @ PsiV
#pragma unroll
    for (int ks = 0; ks < ROWS / 16; ++ks) {
      unsigned a[4], b[4];
      ldsm_x4(a, &S.g[(16 * orow + (lane & 15)) * LDG + 16 * ks + 8 * (lane >> 4)]);
      ldsm_x4_t(b, &S.pv[swp(16 * ks + 8 * (q4 & 1) + (lane & 7), 16 * ocol + 8 * (q4 >> 1))]);
      mma(o[0], a, b[0], b[1]);
      mma(o[1], a, b[2], b[3]);
    }
    if (it + 1 < nsb) {
      const float dec = __expf(S.tok[nv - 1][0]);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) st[mt][nt][e] *= dec;
#pragma unroll
      for (int ks = 0; ks < ROWS / 16; ++ks) {
        unsigned a[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
          ldsm_x4_t(a[mt], &S.k[swn(16 * ks + 8 * (q4 >> 1) + (lane & 7), 32 * sr + 16 * mt + 8 * (q4 & 1))]);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          unsigned b[4];
          ldsm_x4_t(b, &S.pv[swp(16 * ks + 8 * (q4 & 1) + (lane & 7), 32 * sc + 16 * np + 8 * (q4 >> 1))]);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma(st[mt][2 * np], a[mt], b[0], b[1]);
            mma(st[mt][2 * np + 1], a[mt], b[2], b[3]);
          }
        }
      }
    }
    // ---- D skip, gate, sum over the R ranks (rows r .. r + 3 sit in lanes 4 apart)
    {
      const int rank = g_ & 3;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int p0 = 16 * ocol + 8 * nt + 2 * cq;
#pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          const int row = ri + 8 * eh;
          const float2 pv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&S.pv[swp(row, p0)]));
          const float2 gz = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&S.gz[swp(row, p0)]));
          float v0 = (o[nt][2 * eh] + dsk * pv.x) * gz.x;
          float v1 = (o[nt][2 * eh + 1] + dsk * pv.y) * gz.y;
          v0 += __shfl_xor_sync(0xffffffffu, v0, 4);
          v1 += __shfl_xor_sync(0xffffffffu, v1, 4);
          v0 += __shfl_xor_sync(0xffffffffu, v0, 8);
          v1 += __shfl_xor_sync(0xffffffffu, v1, 8);
          const int tk = row >> 2;
          if (rank == 0 && tk < nv)
            *reinterpret_cast<__nv_bfloat162*>(out + ((long)(t0 + tk) * H + h) * P + p0) = __floats2bfloat162_rn(v0, v1);
        }
      }
    }
    if (it + 1 < nsb) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int eh = 0; eh < 2; ++eh) {
            const int n = 32 * sr + 16 * mt + g_ + 8 * eh, p = 32 * sc + 8 * nt + 2 * cq;
            *reinterpret_cast<unsigned*>(&S.sb[swp(n, p)]) = pack_bf16(st[mt][nt][2 * eh], st[mt][nt][2 * eh + 1]);
          }
    }
  }
}
}  // namespace

void mamba3_scan_prefill_sm80(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Z,
                              torch::Tensor qbias, torch::Tensor kbias, torch::Tensor psi, torch::Tensor phi,
                              torch::Tensor zeta, torch::Tensor Dskip, torch::Tensor thl, torch::Tensor ain,
                              torch::Tensor dcs8, torch::Tensor dcsr8, torch::Tensor ts, torch::Tensor gam,
                              torch::Tensor ch_start, torch::Tensor ch_len, torch::Tensor ch_last, torch::Tensor sin_t,
                              torch::Tensor out, torch::Tensor fin_k, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && Z.size(2) == P && Q.size(-1) == N && Q.size(-2) == R && thl.size(-1) == NQ);
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && out.is_contiguous() && sin_t.is_contiguous());
  TORCH_CHECK(V.stride(2) == 1 && Z.stride(2) == 1 && V.scalar_type() == at::kBFloat16 && Z.scalar_type() == at::kBFloat16);
  if (NC * H == 0) return;
  const int smem = (int)sizeof(Smem);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(scan_sm80_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(scan_sm80_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    attr = true;
  }
  scan_sm80_kernel<<<(int)(NC * H), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr()),
      qbias.data_ptr<float>(), kbias.data_ptr<float>(), psi.data_ptr<float>(), phi.data_ptr<float>(),
      zeta.data_ptr<float>(), Dskip.data_ptr<float>(), thl.data_ptr<float>(), ain.data_ptr<float>(),
      dcs8.data_ptr<float>(), dcsr8.data_ptr<float>(), ts.data_ptr<float>(), gam.data_ptr<float>(),
      ch_start.data_ptr<int>(), ch_len.data_ptr<int>(), ch_last.data_ptr<int>(), sin_t.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(fin_k.data_ptr()), T, H,
      V.stride(0), Z.stride(0), V.stride(1), Z.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scan", &mamba3_scan_prefill_sm80); }
