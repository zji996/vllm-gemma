from __future__ import annotations

import math
from functools import lru_cache

import torch


def _normal_pdf(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_sq = sigma * sigma
    return torch.exp(-(x * x) / (2 * sigma_sq)) / math.sqrt(2 * math.pi * sigma_sq)


@lru_cache(maxsize=16)
def solve_lloyd_max_codebook(dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    levels = 2**bits
    sigma = 1.0 / math.sqrt(float(dim))
    lo = -4.5 * sigma
    hi = 4.5 * sigma
    grid = torch.linspace(lo, hi, steps=32769, dtype=torch.float64)
    pdf = _normal_pdf(grid, sigma)

    centroids = torch.linspace(
        lo + (hi - lo) / (2 * levels),
        hi - (hi - lo) / (2 * levels),
        steps=levels,
        dtype=torch.float64,
    )

    for _ in range(64):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        new_centroids = []
        for idx in range(levels):
            left = lo if idx == 0 else boundaries[idx - 1].item()
            right = hi if idx == levels - 1 else boundaries[idx].item()
            mask = (grid >= left) & (grid < right)
            if idx == levels - 1:
                mask = (grid >= left) & (grid <= right)
            weights = pdf[mask]
            values = grid[mask]
            if weights.numel() == 0 or weights.sum().item() == 0.0:
                new_centroids.append(centroids[idx])
                continue
            new_centroids.append((values * weights).sum() / weights.sum())
        updated = torch.stack(new_centroids)
        if torch.max(torch.abs(updated - centroids)).item() < 1e-8:
            centroids = updated
            break
        centroids = updated

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.to(torch.float32), boundaries.to(torch.float32)
