# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.attention.ops.vit_attn_wrappers import vit_torch_sdpa_wrapper

logger = init_logger(__name__)

MAX_BOOL_MASK_BYTES = int(
    os.getenv("VLLM_SPARSE_MASK_MAX_BOOL_BYTES", str(512 * 1024 * 1024))
)

try:
    import sparse_attn_cuda

    HAS_SPARSE_ATTN_CUDA = True
except ImportError:
    sparse_attn_cuda = None
    HAS_SPARSE_ATTN_CUDA = False


def _fallback_to_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    enable_gqa: bool,
    reason: str,
) -> torch.Tensor:
    logger.info_once("SPARSE_MASK backend falling back to torch SDPA: %s", reason)
    if not query.is_cuda:
        valid_tokens = _build_valid_token_mask(
            batch_size=query.shape[0],
            seq_len=query.shape[1],
            cu_seqlens=cu_seqlens,
            device=query.device,
        )
        mask = valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]
        attn_bias = torch.zeros(
            (query.shape[0], 1, query.shape[1], key.shape[1]),
            dtype=query.dtype,
            device=query.device,
        )
        attn_bias.masked_fill_(~mask, torch.finfo(query.dtype).min)

        query_bhnd = query.transpose(1, 2)
        key_bhnd = key.transpose(1, 2)
        value_bhnd = value.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query_bhnd,
            key_bhnd,
            value_bhnd,
            attn_mask=attn_bias,
            scale=scale,
        ).transpose(1, 2)
        return output * valid_tokens[:, :, None, None].to(dtype=output.dtype)

    return vit_torch_sdpa_wrapper(
        q=query,
        k=key,
        v=value,
        scale=scale,
        cu_seqlens=cu_seqlens,
        enable_gqa=enable_gqa,
    )


def _build_valid_token_mask(
    batch_size: int,
    seq_len: int,
    cu_seqlens: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if cu_seqlens is None:
        return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int64).flatten()
    expected = batch_size + 1
    if cu_seqlens.numel() != expected:
        raise ValueError(
            "cu_seqlens must have batch_size + 1 elements for SPARSE_MASK "
            f"(got {cu_seqlens.numel()}, expected {expected})."
        )

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).clamp(min=0, max=seq_len)
    positions = torch.arange(seq_len, device=device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def sparse_mask_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Sparse-mask ViT attention with an SDPA fallback for unsupported cases.

    The CUDA extension expects tensors in [B, H, N, D] layout and an explicit
    boolean mask. vLLM mm-encoder attention uses [B, N, H, D] with optional
    cu_seqlens, so we bridge the two formats here.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return _fallback_to_sdpa(
            query, key, value, scale, cu_seqlens, enable_gqa, "expected 4D q/k/v"
        )

    if query.shape[2] != key.shape[2] or key.shape[2] != value.shape[2]:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            "grouped-query attention is not supported by sparse_attn_cuda",
        )

    if enable_gqa:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            "enable_gqa requires the native SDPA path",
        )

    if not HAS_SPARSE_ATTN_CUDA:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            "sparse_attn_cuda is unavailable",
        )

    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            "CUDA tensors are required",
        )

    if query.dtype not in (torch.float16, torch.bfloat16):
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            f"unsupported dtype {query.dtype}",
        )

    if query.shape[-1] != 64:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            f"head_dim={query.shape[-1]} is unsupported by sparse_attn_cuda",
        )

    batch_size, seq_len, num_heads, _ = query.shape
    if cu_seqlens is not None and cu_seqlens.numel() != batch_size + 1:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            "unsupported cu_seqlens layout "
            f"(got {cu_seqlens.numel()} elements for batch_size={batch_size})",
        )

    dense_mask_bytes = batch_size * num_heads * seq_len * seq_len
    if dense_mask_bytes > MAX_BOOL_MASK_BYTES:
        return _fallback_to_sdpa(
            query,
            key,
            value,
            scale,
            cu_seqlens,
            enable_gqa,
            "dense bool mask would require "
            f"{dense_mask_bytes / (1024**3):.2f} GiB > "
            f"{MAX_BOOL_MASK_BYTES / (1024**3):.2f} GiB budget",
        )

    valid_tokens = _build_valid_token_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        cu_seqlens=cu_seqlens,
        device=query.device,
    )
    attention_mask = (
        valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]
    ).expand(batch_size, num_heads, seq_len, seq_len).contiguous()

    query_bhnd = query.transpose(1, 2).contiguous()
    key_bhnd = key.transpose(1, 2).contiguous()
    value_bhnd = value.transpose(1, 2).contiguous()

    output = torch.empty_like(query_bhnd)
    lse = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        device=query.device,
        dtype=torch.float32,
    )
    packed_mask = sparse_attn_cuda.pack_mask(attention_mask)
    sparse_attn_cuda.forward(
        query_bhnd,
        key_bhnd,
        value_bhnd,
        packed_mask,
        output,
        lse,
        float(scale),
        False,
    )

    output = output.transpose(1, 2).contiguous()
    return output * valid_tokens[:, :, None, None].to(dtype=output.dtype)
