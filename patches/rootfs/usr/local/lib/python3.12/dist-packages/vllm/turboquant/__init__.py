from .config import TurboQuantConfig, get_turboquant_config


def build_turboquant_runtime(*args, **kwargs):
    from .attention import build_turboquant_runtime as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "TurboQuantConfig",
    "get_turboquant_config",
    "build_turboquant_runtime",
]
