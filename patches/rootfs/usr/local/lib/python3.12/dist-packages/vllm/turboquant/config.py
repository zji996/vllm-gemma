from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_layers(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return ()
    layers: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layers.append(int(chunk))
    return tuple(sorted(set(layers)))


@dataclass(frozen=True)
class TurboQuantConfig:
    enabled: bool
    bits: int
    target: str
    mode: str
    sample_layers: tuple[int, ...]
    log_file: str
    max_shadow_records: int
    max_shadow_seq_len: int
    max_shadow_query_tokens: int
    seed: int
    qjl_dim: int

    @property
    def shadow_enabled(self) -> bool:
        return self.enabled and self.mode == "shadow"

    @property
    def runtime_enabled(self) -> bool:
        return self.enabled and self.mode == "runtime"

    def layer_enabled(self, layer_idx: int) -> bool:
        if not self.enabled or self.target != "full_attention":
            return False
        return not self.sample_layers or layer_idx in self.sample_layers

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def get_turboquant_config() -> TurboQuantConfig:
    enabled = _parse_bool(os.getenv("VLLM_TURBOQUANT_ENABLE"), False)
    bits = int(os.getenv("VLLM_TURBOQUANT_BITS", "4"))
    if bits not in {2, 3, 4}:
        raise ValueError(f"Unsupported VLLM_TURBOQUANT_BITS={bits}; expected one of 2,3,4")
    target = os.getenv("VLLM_TURBOQUANT_TARGET", "full_attention").strip() or "full_attention"
    mode = os.getenv("VLLM_TURBOQUANT_MODE", "off").strip().lower() or "off"
    if not enabled or mode == "off":
        enabled = False
        mode = "off"
    if mode not in {"off", "shadow", "runtime"}:
        raise ValueError(f"Unsupported VLLM_TURBOQUANT_MODE={mode}")

    log_file = os.getenv(
        "VLLM_TURBOQUANT_LOG_FILE",
        str(Path("/root/.cache/vllm/turboquant") / "shadow_metrics.jsonl"),
    )

    return TurboQuantConfig(
        enabled=enabled,
        bits=bits,
        target=target,
        mode=mode,
        sample_layers=_parse_layers(os.getenv("VLLM_TURBOQUANT_SAMPLE_LAYERS")),
        log_file=log_file,
        max_shadow_records=max(int(os.getenv("VLLM_TURBOQUANT_MAX_SHADOW_RECORDS", "8")), 0),
        max_shadow_seq_len=max(int(os.getenv("VLLM_TURBOQUANT_MAX_SHADOW_SEQ_LEN", "512")), 1),
        max_shadow_query_tokens=max(int(os.getenv("VLLM_TURBOQUANT_MAX_SHADOW_QUERY_TOKENS", "2")), 1),
        seed=int(os.getenv("VLLM_TURBOQUANT_SEED", "20260326")),
        qjl_dim=max(int(os.getenv("VLLM_TURBOQUANT_QJL_DIM", "0")), 0),
    )
