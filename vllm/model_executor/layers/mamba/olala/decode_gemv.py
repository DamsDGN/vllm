# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token GEMV for Olala's decode projections.

cuBLAS's batch-1 kernels sit on a ~4.2 us floor whatever the matrix size,
so the many small projections of a decode step (latent MoE, in_proj_dyn,
TPA factors) run at well under 1 TB/s. This streaming kernel — each program
owns BN output rows and walks K in BK chunks — is 2.5-3 us on those shapes
and also ahead of cuBLAS on the large ones (measured under CUDA graphs on
GH200). For 2..64 rows the skinny kernel puts the weight tile on the MMA's M
side so every weight byte is still read once (cuBLAS picks 55-95 % of the
copy bandwidth there on an A4500, the skinny kernel 80-97 %).
"""

import os
from functools import cache

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


_DISABLED = os.environ.get("OLALA_GEMV", "1") == "0"


@triton.jit
def _gemv_m1_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        x = tl.load(x_ptr + offs_k)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _gemv_m1_full_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """Whole (padded) K extent per program: every load is in flight at once
    instead of walking K in a dependent loop."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    k_mask = offs_k < K
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
        mask=n_mask[:, None] & k_mask[None, :], other=0.0,
    )
    x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)
    acc = tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


@cache
def is_blackwell() -> bool:
    return torch.cuda.get_device_capability()[0] >= 10


def _decode_gemv_blackwell(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """B300 (weights streamed from HBM, CUDA graphs): the full-K kernel is
    2.7 us on the latent MoE projections where the K loop takes 4-12 us, and
    3.8 / 5.1 us on mixer_proj (K = 3072 / 4608) vs 4.5 / 9.0 us. Wide outputs
    (N >= 4096: in_proj, the TPA projections) are faster on cuBLAS."""
    N, K = weight.shape
    bk = triton.next_power_of_2(K)
    if N >= 4096 or bk > 8192:
        return F.linear(x, weight)
    if bk <= 2048:
        bn, nw = 2, 8
    elif bk <= 4096:
        bn, nw = 1, 4
    else:
        bn, nw = 2, 2
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    _gemv_m1_full_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, N, K, weight.stride(0), BN=bn, BK=bk, num_warps=nw,
    )
    return y


@triton.jit
def _skinny_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """``y = x @ w.T`` for a few rows of x: the BN weight rows are the MMA's M side and the
    BM <= 64 rows of x its N side, so every weight byte is read once."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    n_ok = offs_n[:, None] < N
    m_ok = offs_m[None, :] < M
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for _ in range(0, K, BK):
        w = tl.load(w_ptrs, mask=n_ok, other=0.0)
        x = tl.load(x_ptrs, mask=m_ok, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptrs += BK
        x_ptrs += BK
    y_ptrs = y_ptr + offs_m[None, :] * stride_ym + offs_n[:, None]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_ok & m_ok)


@triton.jit
def _skinny_q8_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym, stride_sn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    """``_skinny_kernel`` with int8 weights: per-row scales (GROUP == 0) or per row and
    GROUP-column group (BK == GROUP); int8 -> bf16 is exact, accumulation in fp32."""
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    n_ok = offs_n[:, None] < N
    m_ok = offs_m[None, :] < M
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for kb in range(0, K // BK):
        w = tl.load(w_ptrs, mask=n_ok, other=0).to(tl.bfloat16)
        x = tl.load(x_ptrs, mask=m_ok, other=0.0)
        if GROUP == 0:
            acc = tl.dot(w, x, acc)
        else:
            s = tl.load(s_ptr + offs_n * stride_sn + kb, mask=offs_n < N, other=0.0)
            acc += tl.dot(w, x) * s[:, None]
        w_ptrs += BK
        x_ptrs += BK
    if GROUP == 0:
        s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc * s[:, None]
    y_ptrs = y_ptr + offs_m[None, :] * stride_ym + offs_n[:, None]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_ok & m_ok)


@triton.jit
def _gemv_m1_q8_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, N, K, stride_wn, stride_sn,
    BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        if GROUP == 0:
            acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
        else:
            s = tl.load(s_ptr + offs_n[:, None] * stride_sn + offs_k[None, :] // GROUP,
                        mask=n_mask[:, None], other=0.0)
            acc += tl.sum(w.to(tl.float32) * s * x[None, :], axis=1)
    if GROUP == 0:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


# Int8 W8A16 copies of the dense decode weights (OLALA_DENSE_INT8=row|g128), keyed by the bf16
# tensor they replace: (data_ptr, rows, cols) -> (int8 [N, K], fp32 scales [N] or [N, K // G]).
_Q8: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_Q8_GROUP = {"row": 0, "g128": 128}.get(os.environ.get("OLALA_DENSE_INT8", ""), None)
_Q8_MISSES: set[tuple[int, int]] = set()
# Above 32 rows the int8 kernel is no faster than the bf16 one (dequantized weights, same values).
Q8_MAX_M = int(os.environ.get("OLALA_Q8_MAX_M", "32"))


def quantize_int8(w: torch.Tensor, group: int, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 of a (N, K) weight: per row (group 0) or per row and group columns
    (scales [N] or [N, K // group], fp32); row chunks keep the fp32 temporaries small."""
    N, K = w.shape
    g = group if group and K % group == 0 else K
    q = torch.empty(N, K, dtype=torch.int8, device=w.device)
    s = torch.empty(N, K // g, dtype=torch.float32, device=w.device)
    for r in range(0, N, chunk):
        v = w[r : r + chunk].float().view(-1, K // g, g)
        sc = v.abs().amax(-1).clamp_min(1e-12) / 127.0
        q[r : r + chunk] = (v / sc[..., None]).round().clamp(-127, 127).to(torch.int8).view(-1, K)
        s[r : r + chunk] = sc
    return q, (s.squeeze(-1).contiguous() if g == K else s)


def fake_quant_(w: torch.Tensor, group: int, chunk: int = 8192) -> None:
    """In place: w <- dequant(quant(w)) (what a W8A16 kernel computes with), for the prefill GEMMs."""
    N, K = w.shape
    g = group if group and K % group == 0 else K
    for r in range(0, N, chunk):
        v = w[r : r + chunk].float().view(-1, K // g, g)
        sc = v.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
        w[r : r + chunk] = ((v / sc).round().clamp(-127, 127) * sc).view(-1, K).to(w.dtype)


def register_int8(w: torch.Tensor, group: int) -> None:
    """Int8 copy of a decode weight; a row slice of an already registered buffer reuses its rows."""
    key = (w.data_ptr(), w.shape[0], w.shape[1])
    if key in _Q8 or w.dim() != 2 or w.stride(-1) != 1:
        return
    for (ptr, n, k), (q, s) in list(_Q8.items()):
        off = w.data_ptr() - ptr
        row = w.stride(0) * w.element_size()
        if k == w.shape[1] and 0 <= off and off % row == 0 and off // row + w.shape[0] <= n:
            r0 = off // row
            _Q8[key] = (q[r0 : r0 + w.shape[0]], s[r0 : r0 + w.shape[0]])
            return
    _Q8[key] = quantize_int8(w, group)


def _q8_lookup(weight: torch.Tensor):
    hit = _Q8.get((weight.data_ptr(), weight.shape[0], weight.shape[1]))
    if hit is None and (weight.shape[0], weight.shape[1]) not in _Q8_MISSES:
        _Q8_MISSES.add((weight.shape[0], weight.shape[1]))
        import logging
        logging.getLogger(__name__).warning("OLALA_DENSE_INT8: no int8 copy for a %s decode weight",
                                            tuple(weight.shape))
    return hit


def skinny_linear_q8(
    x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, cfg: tuple[int, int, int, int] | None = None
) -> torch.Tensor:
    M, K = x.shape
    N = q.shape[0]
    group = 0 if s.dim() == 1 else K // s.shape[1]
    bn, bk, stages, warps = cfg or _skinny_q8_cfg(M, N, K)
    if group:
        bk = group
    elif K % bk:
        bk = 128
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _skinny_q8_kernel[(triton.cdiv(N, bn),)](
        x, q, s, y, M, N, K, x.stride(0), q.stride(0), y.stride(0), s.stride(0),
        BM=16 if M <= 16 else (32 if M <= 32 else 64), BN=bn, BK=bk, GROUP=group,
        num_stages=stages, num_warps=warps,
    )
    return y


_Q8_CFG = os.environ.get("OLALA_Q8_CFG")  # "bn,bk,stages,warps" override (tuning)


def _skinny_q8_cfg(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    # A4500 sweep (graphed, weights from DRAM), M <= 16 vs the bf16 kernel: in_proj / TPA cat /
    # gate_proj 1.8x, lm_head 1.95x, MoE w_in 1.45x (64x128); N = 1536 (mixer_proj, MoE w_out)
    # 1.25-1.55x (32x256); at 32 rows 1.1-1.6x.
    if _Q8_CFG:
        return tuple(int(v) for v in _Q8_CFG.split(","))  # type: ignore[return-value]
    if N > 1536:
        return 64, 128, 3, 4
    return 32, (256 if K % 256 == 0 else 128), 4, 4


def gemv_m1_q8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    group = 0 if s.dim() == 1 else K // s.shape[1]
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    bn, bk, nw = (8, 512, 4) if N >= 4096 else (4, 512 if K % 512 == 0 else 128, 8)
    _gemv_m1_q8_kernel[(triton.cdiv(N, bn),)](
        x, q, s, y, N, K, q.stride(0), s.stride(0), BN=bn, BK=bk, GROUP=group, num_warps=nw,
    )
    return y


def skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, cfg: tuple[int, int, int, int] | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for M <= 64 rows (K % BK == 0); ``cfg`` = (BN, BK, stages, warps)."""
    M, K = x.shape
    N = weight.shape[0]
    bn, bk, stages, warps = cfg or _skinny_cfg(M, N, K)
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _skinny_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, M, N, K, x.stride(0), weight.stride(0), y.stride(0),
        BM=16 if M <= 16 else (32 if M <= 32 else 64), BN=bn, BK=bk,
        num_stages=stages, num_warps=warps,
    )
    return y


def _skinny_cfg(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    # Sweep of the decode shapes on an RTX A4500 (graphed, weights from DRAM): 88-97 % of the
    # copy bandwidth up to 32 rows, 80-88 % at 64; split-K never won.
    if M > 32 and N > 1536 and K < 3072:
        return 64, 64, 3, 4
    return 32, 128, 4, (8 if M > 32 else 4)


SKINNY_MAX_M = 64
_SKINNY = os.environ.get("OLALA_SKINNY", "1") == "1"


def decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for ``x: (M, K)``, ``weight: (N, K)``: the skinny kernel up to
    SKINNY_MAX_M rows (bf16), except single rows with K <= 1536 (triton GEMV, as fast there);
    F.linear otherwise. At one row the GEMV's 128-wide K blocks are slow for K = 1920 (w_out:
    53 % of the copy bandwidth vs 89 % on an A4500). Blackwell keeps its own dispatch: the full-K
    GEMV at one row, F.linear otherwise."""
    M = x.shape[0]
    if x.is_cuda and is_blackwell():
        if M != 1 or x.dtype != weight.dtype or _DISABLED:
            return F.linear(x, weight)
        return _decode_gemv_blackwell(x, weight)
    N, K = weight.shape
    if _DISABLED or not x.is_cuda or x.dtype != weight.dtype or K % 128 != 0 or M > SKINNY_MAX_M:
        return F.linear(x, weight)
    if _Q8_GROUP is not None and M <= Q8_MAX_M and x.dtype == torch.bfloat16 and x.stride(-1) == 1:
        hit = _q8_lookup(weight)
        if hit is not None:
            if M == 1 and K <= 1536:
                return gemv_m1_q8(x, *hit)
            return skinny_linear_q8(x, *hit)
    if (
        (M != 1 or K > 1536) and _SKINNY and x.dtype == torch.bfloat16
        and x.stride(-1) == 1 and weight.stride(-1) == 1
    ):
        return skinny_linear(x, weight)
    if M != 1:
        return F.linear(x, weight)
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    # HBM-streaming configs measured on GH200 (36 distinct weights, graphed):
    # wide outputs like 8 rows/program; narrow outputs want more programs.
    if N >= 4096:
        bn, bk, nw = 8, 512, 4
    elif K % 1024 == 0 and K >= 3072:
        bn, bk, nw = 4, 1024, 8
    else:
        bn, bk, nw = 4, (512 if K % 512 == 0 else 128), 8
    _gemv_m1_kernel[(triton.cdiv(N, bn),)](
        x, weight, y, N, K, weight.stride(0), BN=bn, BK=bk, num_warps=nw,
    )
    return y


def _olala_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return decode_gemv(x, weight)


def _olala_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


# Opaque to torch.compile so the M == 1 dispatch is re-evaluated per cudagraph
# capture instead of being specialized by the first (large) compile.
direct_register_custom_op(
    op_name="olala_linear",
    op_func=_olala_linear,
    mutates_args=[],
    fake_impl=_olala_linear_fake,
)


def olala_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.olala_linear(x, weight)


def cat_rows_shared(params: list[torch.nn.Parameter], pad_to: int = 1, align: int = 1) -> torch.Tensor:
    """Row-concatenate ``params`` and rebind each one to its slice of the result.

    The concatenated decode weight then costs no extra memory (a separate copy
    is ~0.8 GB over the 36 layers, a fifth of the KV budget on a 20 GB GPU).
    Row slices of a contiguous tensor keep the parameters' shapes and strides,
    and in-place weight reloads (``param.data.copy_``) land in the shared
    storage, so the concatenation never goes stale.

    With ``pad_to`` > 1 the buffer gets zero rows up to a multiple of ``pad_to`` (the returned
    tensor includes them): cuBLAS picks Hopper kernels only for aligned output widths. With
    ``align`` > 1 each parameter starts at a multiple of ``align`` rows (zero rows in between),
    so the output columns of every parameter start aligned too.
    """
    with torch.no_grad():
        offs, n_rows = [], 0
        for p in params:
            n_rows = -(-n_rows // align) * align
            offs.append(n_rows)
            n_rows += p.shape[0]
        n_pad = -(-n_rows // pad_to) * pad_to
        ref = params[0].data
        w = torch.zeros(n_pad, *ref.shape[1:], dtype=ref.dtype, device=ref.device)
        for p, off in zip(params, offs):
            n = p.shape[0]
            w[off : off + n].copy_(p.data)
            p.data = w[off : off + n]
    return w
