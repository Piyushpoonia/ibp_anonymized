from __future__ import annotations

from dataclasses import asdict, dataclass


SPATIAL_PREDICATES = (
    "left_of",
    "in_front_of",
    "near",
    "overlapping",
    "occluding",
)

TEMPORAL_PREDICATES = (
    "approaching",
    "moving_away",
    "same_motion_direction",
)

EXPLORATORY_PREDICATES = ("crossing_path",)

OBJECT_CLASSES = (
    "building",
    "garage",
    "car",
    "truck",
    "trailer",
    "caravan",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "bigPole",
    "smallPole",
    "trafficLight",
    "trafficSign",
    "lamp",
    "trashbin",
    "vendingmachine",
    "box",
    "stop",
    "bridge",
    "tunnel",
    "train",
    "bus",
    "unknownConstruction",
    "unknownVehicle",
    "unknownObject",
)

SENSOR_MODALITIES = ("rgb", "lidar", "text")


@dataclass(frozen=True)
class ModelConfig:
    schema_version: str = "IBP-K360-model-v1.3.0"
    dim: int = 384
    rgb_input_dim: int = 768
    lidar_input_dim: int = 512
    text_input_dim: int = 512
    anchor_input_dim: int = 512
    num_parts: int = 8
    num_heads: int = 8
    fusion_layers: int = 2
    temporal_layers: int = 2
    dropout: float = 0.1
    association_geometry_dim: int = 16
    spatial_geometry_dim: int = 16
    temporal_geometry_dim: int = 24
    tracklet_length: int = 5
    num_object_classes: int = len(OBJECT_CLASSES)
    num_spatial_predicates: int = len(SPATIAL_PREDICATES)
    num_temporal_predicates: int = len(TEMPORAL_PREDICATES)
    sinkhorn_iterations: int = 8
    sinkhorn_temperature: float = 0.10
    enabled_modalities: tuple[str, ...] = SENSOR_MODALITIES
    use_parts: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_protocol(config: ModelConfig) -> None:
    if not isinstance(config.use_parts, bool):
        raise ValueError("use_parts must be a boolean checkpointed model setting.")
    if config.dim % config.num_heads:
        raise ValueError("The common feature dimension must be divisible by num_heads.")
    if config.num_parts != 8:
        raise ValueError("IBP-K360-v1 freezes the number of latent parts at eight.")
    if config.tracklet_length != 5:
        raise ValueError("IBP-K360-v1 freezes tracklets at five key samples.")
    if config.num_object_classes != len(OBJECT_CLASSES):
        raise ValueError(
            f"The frozen ontology has {len(OBJECT_CLASSES)} object classes, "
            f"but the model declares {config.num_object_classes}."
        )
    if config.num_spatial_predicates != 5 or config.num_temporal_predicates != 3:
        raise ValueError("The active protocol trains five spatial and three temporal predicates.")
    enabled = tuple(config.enabled_modalities)
    if not enabled:
        raise ValueError("At least one sensor modality must be enabled.")
    unknown = sorted(set(enabled) - set(SENSOR_MODALITIES))
    if unknown:
        raise ValueError(f"Unknown sensor modalities: {unknown}")
    canonical = tuple(name for name in SENSOR_MODALITIES if name in enabled)
    if enabled != canonical:
        raise ValueError(
            f"Enabled modalities must be unique and use canonical order {SENSOR_MODALITIES}."
        )
