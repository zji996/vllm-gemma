from __future__ import annotations

import math

import torch

from .codebook import solve_lloyd_max_codebook
from .rotation import generate_qjl_matrix, generate_rotation_matrix


class TurboQuantKeyCompressor:
    def __init__(
        self,
        head_dim: int,
        bits: int,
        *,
        seed: int,
        device: torch.device,
        qjl_dim: int = 0,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or head_dim
        self.device = device

        self.rotation = generate_rotation_matrix(head_dim, seed, device=device)
        self.qjl = generate_qjl_matrix(head_dim, self.qjl_dim, seed + 10_000, device=device)
        centroids, boundaries = solve_lloyd_max_codebook(head_dim, self.mse_bits)
        self.centroids = centroids.to(device=device)
        self.boundaries = boundaries.to(device=device)

    def _quantize(self, rotated: torch.Tensor) -> torch.Tensor:
        flat = rotated.reshape(-1, self.head_dim)
        indices = torch.bucketize(flat, self.boundaries)
        return indices.clamp_max(self.centroids.numel() - 1).to(torch.uint8).reshape(rotated.shape)

    def _dequantize_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids[indices.long()]

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        states_f = states.float()
        vec_norms = torch.linalg.norm(states_f, dim=-1, keepdim=True)
        normalized = states_f / (vec_norms + 1e-8)
        rotated = normalized @ self.rotation.T
        indices = self._quantize(rotated)
        rotated_mse = self._dequantize_indices(indices)
        mse = (rotated_mse @ self.rotation) * vec_norms
        residual = states_f - mse
        residual_norm = torch.linalg.norm(residual, dim=-1)
        qjl_projected = residual @ self.qjl.T
        qjl_signs = torch.where(qjl_projected >= 0, 1, -1).to(torch.int8)
        return {
            "indices": indices,
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "residual_norm": residual_norm.to(torch.float16),
            "qjl_signs": qjl_signs,
            "shape": tuple(states.shape),
        }

    @torch.no_grad()
    def decompress_mse(self, compressed: dict) -> torch.Tensor:
        rotated_mse = self._dequantize_indices(compressed["indices"])
        norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (rotated_mse @ self.rotation) * norms

    @torch.no_grad()
    def asymmetric_attention_scores(
        self,
        queries: torch.Tensor,
        compressed: dict,
    ) -> torch.Tensor:
        k_mse = self.decompress_mse(compressed)
        qjl_signs = compressed["qjl_signs"].float()
        residual_norm = compressed["residual_norm"].float().unsqueeze(-2)

        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        q_projected = torch.matmul(queries.float(), self.qjl.T)
        qjl_inner = torch.matmul(q_projected, qjl_signs.transpose(-2, -1))
        correction = math.sqrt(math.pi / 2.0) / float(self.qjl_dim)
        return term1 + correction * qjl_inner * residual_norm


class TurboQuantValueCompressor:
    def __init__(
        self,
        head_dim: int,
        bits: int,
        *,
        seed: int,
        device: torch.device,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.rotation = generate_rotation_matrix(head_dim, seed, device=device)
        centroids, boundaries = solve_lloyd_max_codebook(head_dim, bits)
        self.centroids = centroids.to(device=device)
        self.boundaries = boundaries.to(device=device)

    def _quantize(self, rotated: torch.Tensor) -> torch.Tensor:
        flat = rotated.reshape(-1, self.head_dim)
        indices = torch.bucketize(flat, self.boundaries)
        return indices.clamp_max(self.centroids.numel() - 1).to(torch.uint8).reshape(rotated.shape)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        states_f = states.float()
        vec_norms = torch.linalg.norm(states_f, dim=-1, keepdim=True)
        normalized = states_f / (vec_norms + 1e-8)
        rotated = normalized @ self.rotation.T
        indices = self._quantize(rotated)
        return {
            "indices": indices,
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "shape": tuple(states.shape),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        rotated = self.centroids[compressed["indices"].long()]
        norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (rotated @ self.rotation) * norms
