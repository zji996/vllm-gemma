from __future__ import annotations

import torch


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def generate_rotation_matrix(
    dim: int,
    seed: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    gaussian = torch.randn(
        dim,
        dim,
        generator=_cpu_generator(seed),
        dtype=torch.float32,
        device="cpu",
    )
    q_mat, r_mat = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r_mat))
    diag_sign[diag_sign == 0] = 1.0
    return (q_mat * diag_sign.unsqueeze(0)).to(device=device)


def generate_qjl_matrix(
    dim: int,
    qjl_dim: int,
    seed: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    matrix = torch.randn(
        qjl_dim,
        dim,
        generator=_cpu_generator(seed),
        dtype=torch.float32,
        device="cpu",
    )
    return matrix.to(device=device)
