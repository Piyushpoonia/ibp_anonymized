"""Part-aware temporal scene-graph model for the KITTI-360 IBP protocol."""

from .protocol import ModelConfig, SPATIAL_PREDICATES, TEMPORAL_PREDICATES

__all__ = [
    "IBPK360Model",
    "ModelConfig",
    "SPATIAL_PREDICATES",
    "TEMPORAL_PREDICATES",
]


def __getattr__(name: str):
    if name == "IBPK360Model":
        from .model import IBPK360Model

        return IBPK360Model
    raise AttributeError(name)
