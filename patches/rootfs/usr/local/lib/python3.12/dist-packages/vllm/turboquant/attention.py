from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import solve_lloyd_max_codebook
from .config import get_turboquant_config
from .rotation import generate_qjl_matrix, generate_rotation_matrix

logger = logging.getLogger(__name__)


def _disable_dynamo(fn):
    dynamo = getattr(torch, "_dynamo", None)
    disable = getattr(dynamo, "disable", None)
    if disable is None:
        return fn
    return disable(fn)


def _reshape_tokens_to_bhsd(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    num_tokens = tensor.shape[0]
    return tensor.reshape(num_tokens, num_heads, head_dim).permute(1, 0, 2).unsqueeze(0)


def _reshape_bhsd_to_tokens(tensor: torch.Tensor) -> torch.Tensor:
    squeezed = tensor.squeeze(0).permute(1, 0, 2).contiguous()
    return squeezed.view(squeezed.shape[0], -1)


def _expand_kv_heads(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
    current_heads = tensor.shape[1]
    if current_heads == target_heads:
        return tensor
    repeat = target_heads // current_heads
    return tensor.repeat_interleave(repeat, dim=1)


def _slice_recent_seq(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    if tensor.shape[-2] <= limit:
        return tensor
    return tensor[..., -limit:, :]


def _compression_ratio(bits: int, head_dim: int, qjl_dim: int) -> float:
    mse_bits = max(bits - 1, 1)
    key_bits = head_dim * mse_bits + qjl_dim + 32
    value_bits = head_dim * bits + 16
    raw_bits = 32 * head_dim
    return raw_bits / float(key_bits + value_bits)


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if is_compiling is not None:
            return bool(is_compiling())

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if is_compiling is not None:
            return bool(is_compiling())

    return False


class TurboQuantRuntime(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        prefix: str,
        head_dim: int,
    ) -> None:
        super().__init__()
        cfg = get_turboquant_config()
        self.enabled = cfg.layer_enabled(layer_idx)
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.head_dim = head_dim
        self.bits = cfg.bits
        self.key_bits = max(cfg.bits - 1, 1)
        self.qjl_dim = cfg.qjl_dim or head_dim
        self.shadow_enabled = cfg.shadow_enabled
        self.runtime_enabled = cfg.runtime_enabled
        self.max_shadow_records = cfg.max_shadow_records
        self.max_shadow_seq_len = cfg.max_shadow_seq_len
        self.max_shadow_query_tokens = cfg.max_shadow_query_tokens
        self.log_file = cfg.log_file
        self.shadow_records_emitted = 0

        if not self.enabled:
            self.register_buffer("key_rotation", torch.empty(0), persistent=False)
            self.register_buffer("key_qjl", torch.empty(0), persistent=False)
            self.register_buffer("key_centroids", torch.empty(0), persistent=False)
            self.register_buffer("key_boundaries", torch.empty(0), persistent=False)
            self.register_buffer("value_rotation", torch.empty(0), persistent=False)
            self.register_buffer("value_centroids", torch.empty(0), persistent=False)
            self.register_buffer("value_boundaries", torch.empty(0), persistent=False)
            return

        key_centroids, key_boundaries = solve_lloyd_max_codebook(head_dim, self.key_bits)
        value_centroids, value_boundaries = solve_lloyd_max_codebook(head_dim, cfg.bits)
        self.register_buffer(
            "key_rotation",
            generate_rotation_matrix(head_dim, cfg.seed + layer_idx * 1000, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "key_qjl",
            generate_qjl_matrix(
                head_dim,
                self.qjl_dim,
                cfg.seed + layer_idx * 1000 + 10_000,
                device="cpu",
            ),
            persistent=False,
        )
        self.register_buffer("key_centroids", key_centroids, persistent=False)
        self.register_buffer("key_boundaries", key_boundaries, persistent=False)
        self.register_buffer(
            "value_rotation",
            generate_rotation_matrix(head_dim, cfg.seed + layer_idx * 1000 + 500, device="cpu"),
            persistent=False,
        )
        self.register_buffer("value_centroids", value_centroids, persistent=False)
        self.register_buffer("value_boundaries", value_boundaries, persistent=False)
        logger.info(
            "TurboQuant enabled for %s layer=%s bits=%s mode=%s qjl_dim=%s",
            prefix,
            layer_idx,
            self.bits,
            cfg.mode,
            self.qjl_dim,
        )

    def _quantize(
        self,
        rotated: torch.Tensor,
        boundaries: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        flat = rotated.reshape(-1, self.head_dim)
        indices = torch.bucketize(flat, boundaries)
        return indices.clamp_max(centroids.numel() - 1).to(torch.uint8).reshape(rotated.shape)

    def _dequantize(self, indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        return centroids[indices.long()]

    def _compress_keys(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states_f = states.float()
        vec_norms = torch.linalg.norm(states_f, dim=-1, keepdim=True)
        normalized = states_f / (vec_norms + 1e-8)
        rotated = normalized @ self.key_rotation.T
        indices = self._quantize(rotated, self.key_boundaries, self.key_centroids)
        rotated_mse = self._dequantize(indices, self.key_centroids)
        mse = (rotated_mse @ self.key_rotation) * vec_norms
        residual = states_f - mse
        residual_norm = torch.linalg.norm(residual, dim=-1)
        qjl_projected = residual @ self.key_qjl.T
        qjl_signs = torch.where(qjl_projected >= 0, 1, -1).to(torch.int8)
        return indices, vec_norms.squeeze(-1).to(torch.float16), residual_norm.to(torch.float16), qjl_signs

    def _decompress_keys_mse(
        self,
        indices: torch.Tensor,
        vec_norms: torch.Tensor,
    ) -> torch.Tensor:
        rotated_mse = self._dequantize(indices, self.key_centroids)
        norms = vec_norms.float().unsqueeze(-1)
        return (rotated_mse @ self.key_rotation) * norms

    def _asymmetric_attention_scores(
        self,
        queries: torch.Tensor,
        indices: torch.Tensor,
        vec_norms: torch.Tensor,
        residual_norm: torch.Tensor,
        qjl_signs: torch.Tensor,
    ) -> torch.Tensor:
        k_mse = self._decompress_keys_mse(indices, vec_norms)
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        q_projected = torch.matmul(queries.float(), self.key_qjl.T)
        qjl_inner = torch.matmul(q_projected, qjl_signs.float().transpose(-2, -1))
        correction = torch.sqrt(
            torch.tensor(torch.pi / 2.0, device=queries.device, dtype=torch.float32)
        ) / float(self.qjl_dim)
        return term1 + correction * qjl_inner * residual_norm.float().unsqueeze(-2)

    def _compress_values(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states_f = states.float()
        vec_norms = torch.linalg.norm(states_f, dim=-1, keepdim=True)
        normalized = states_f / (vec_norms + 1e-8)
        rotated = normalized @ self.value_rotation.T
        indices = self._quantize(rotated, self.value_boundaries, self.value_centroids)
        return indices, vec_norms.squeeze(-1).to(torch.float16)

    def _decompress_values(
        self,
        indices: torch.Tensor,
        vec_norms: torch.Tensor,
    ) -> torch.Tensor:
        rotated = self._dequantize(indices, self.value_centroids)
        norms = vec_norms.float().unsqueeze(-1)
        return (rotated @ self.value_rotation) * norms

    @_disable_dynamo
    def _maybe_append_shadow_record(
        self,
        *,
        q_bhsd: torch.Tensor,
        k_bhsd: torch.Tensor,
        v_bhsd: torch.Tensor,
        key_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        value_state: tuple[torch.Tensor, torch.Tensor],
        num_heads: int,
        scaling: float,
    ) -> None:
        if self.max_shadow_records == 0 or self.shadow_records_emitted >= self.max_shadow_records:
            return
        self.shadow_records_emitted += 1

        key_indices, key_vec_norms, key_residual_norm, key_qjl_signs = key_state
        value_indices, value_vec_norms = value_state

        # Shadow logging is intentionally kept out of torch.compile graphs.
        sample_q = _slice_recent_seq(q_bhsd.detach().float(), self.max_shadow_query_tokens)
        sample_keys = _slice_recent_seq(k_bhsd.detach().float(), self.max_shadow_seq_len)
        sample_values = _slice_recent_seq(v_bhsd.detach().float(), self.max_shadow_seq_len)
        sample_key_indices = _slice_recent_seq(key_indices, self.max_shadow_seq_len)
        sample_key_vec_norms = _slice_recent_seq(
            key_vec_norms.unsqueeze(-1),
            self.max_shadow_seq_len,
        ).squeeze(-1)
        sample_key_residual_norm = _slice_recent_seq(
            key_residual_norm.unsqueeze(-1),
            self.max_shadow_seq_len,
        ).squeeze(-1)
        sample_key_qjl_signs = _slice_recent_seq(key_qjl_signs, self.max_shadow_seq_len)
        sample_value_indices = _slice_recent_seq(value_indices, self.max_shadow_seq_len)
        sample_value_vec_norms = _slice_recent_seq(
            value_vec_norms.unsqueeze(-1),
            self.max_shadow_seq_len,
        ).squeeze(-1)

        approx_scores = self._asymmetric_attention_scores(
            sample_q,
            sample_key_indices,
            sample_key_vec_norms,
            sample_key_residual_norm,
            sample_key_qjl_signs,
        ) * scaling
        approx_values = self._decompress_values(
            sample_value_indices,
            sample_value_vec_norms,
        )
        real_keys = _expand_kv_heads(sample_keys, num_heads)
        real_scores = torch.matmul(sample_q, real_keys.transpose(-2, -1)) * scaling

        real_rows = real_scores.reshape(-1, real_scores.shape[-1])
        approx_rows = approx_scores.reshape(-1, approx_scores.shape[-1])
        cosines = F.cosine_similarity(real_rows, approx_rows, dim=-1)

        if real_rows.numel():
            top1_match = (
                (real_rows.argmax(dim=-1) == approx_rows.argmax(dim=-1)).float().mean().item()
            )
        else:
            top1_match = 0.0

        topk = min(5, real_rows.shape[-1])
        if topk > 0:
            real_top1 = real_rows.argmax(dim=-1, keepdim=True)
            approx_topk = approx_rows.topk(topk, dim=-1).indices
            top5_match = (approx_topk == real_top1).any(dim=-1).float().mean().item()
        else:
            top5_match = 0.0

        key_mse = self._decompress_keys_mse(sample_key_indices, sample_key_vec_norms)
        key_rel_mse = (
            ((key_mse - sample_keys) ** 2).mean() / (sample_keys.square().mean() + 1e-8)
        ).item()
        value_rel_mse = (
            ((approx_values - sample_values) ** 2).mean()
            / (sample_values.square().mean() + 1e-8)
        ).item()

        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": "turboquant_shadow",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "layer_idx": self.layer_idx,
            "prefix": self.prefix,
            "bits": self.bits,
            "mode": "shadow",
            "num_heads": num_heads,
            "num_kv_heads": k_bhsd.shape[1],
            "q_tokens": int(sample_q.shape[-2]),
            "kv_tokens": int(sample_keys.shape[-2]),
            "head_dim": int(k_bhsd.shape[-1]),
            "device": str(k_bhsd.device),
            "score_cosine": round(float(cosines.mean().item()), 6),
            "top1_match": round(float(top1_match), 6),
            "top5_match": round(float(top5_match), 6),
            "key_rel_mse": round(float(key_rel_mse), 6),
            "value_rel_mse": round(float(value_rel_mse), 6),
            "estimated_kv_compression_ratio": round(
                _compression_ratio(self.bits, int(k_bhsd.shape[-1]), self.qjl_dim),
                4,
            ),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        scaling: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return k, v

        q_bhsd = _reshape_tokens_to_bhsd(q, num_heads, self.head_dim)
        k_bhsd = _reshape_tokens_to_bhsd(k, num_kv_heads, self.head_dim)
        v_bhsd = _reshape_tokens_to_bhsd(v, num_kv_heads, self.head_dim)

        key_state = self._compress_keys(k_bhsd)
        value_state = self._compress_values(v_bhsd)

        if self.shadow_enabled:
            self._maybe_append_shadow_record(
                q_bhsd=q_bhsd,
                k_bhsd=k_bhsd,
                v_bhsd=v_bhsd,
                key_state=key_state,
                value_state=value_state,
                num_heads=num_heads,
                scaling=scaling,
            )

        if not self.runtime_enabled:
            return k, v

        runtime_k = self._decompress_keys_mse(key_state[0], key_state[1]).to(dtype=k.dtype)
        runtime_v = self._decompress_values(value_state[0], value_state[1]).to(dtype=v.dtype)
        return _reshape_bhsd_to_tokens(runtime_k), _reshape_bhsd_to_tokens(runtime_v)


def build_turboquant_runtime(
    *,
    layer_idx: int,
    prefix: str,
    head_dim: int,
) -> TurboQuantRuntime | None:
    runtime = TurboQuantRuntime(
        layer_idx=layer_idx,
        prefix=prefix,
        head_dim=head_dim,
    )
    if not runtime.enabled:
        return None
    return runtime
