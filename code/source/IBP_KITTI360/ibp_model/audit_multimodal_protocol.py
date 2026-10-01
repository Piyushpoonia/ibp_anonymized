from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .protocol import OBJECT_CLASSES, SPATIAL_PREDICATES, TEMPORAL_PREDICATES


SCHEMA_VERSION = "IBP-K360-full-multimodal-audit-v1.0.0"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    errors: list[str] = []
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "strict_full_multimodal",
        "definition": {
            "object": "RGB, LiDAR and caption-text are all available",
            "spatial": "both objects satisfy the object definition",
            "temporal": (
                "both objects jointly satisfy the object definition in at least three of five "
                "steps, including the middle step"
            ),
        },
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        sequences = split[split_name]
        object_counts = np.zeros(len(OBJECT_CLASSES), dtype=np.int64)
        spatial_counts = np.zeros(len(SPATIAL_PREDICATES), dtype=np.int64)
        temporal_counts = np.zeros(len(TEMPORAL_PREDICATES), dtype=np.int64)
        total_objects = full_objects = spatial_examples = temporal_examples = associations = 0
        sequence_rows: dict[str, Any] = {}
        for sequence in sequences:
            feature_path = args.feature_root / f"{sequence}.h5"
            relation_path = args.relation_root / f"{sequence}.h5"
            if not feature_path.is_file() or not relation_path.is_file():
                errors.append(f"Missing feature/relation shard for {sequence}")
                continue
            with h5py.File(feature_path, "r") as features:
                masks = np.asarray(features["modality_mask"], dtype=np.bool_)
                labels = np.asarray(features["category_id"], dtype=np.int64)
                eligible = masks.all(axis=1)
                total_objects += len(labels)
                full_objects += int(eligible.sum())
                object_counts += np.bincount(
                    labels[eligible], minlength=len(OBJECT_CLASSES)
                )[: len(OBJECT_CLASSES)]
            with h5py.File(relation_path, "r") as relations:
                if not bool(relations.attrs.get("full_multimodal_only", False)):
                    errors.append(f"Relation shard is not strict full-multimodal: {sequence}")
                spatial = np.asarray(relations["spatial/targets"], dtype=np.uint8)
                temporal = np.asarray(relations["temporal/targets"], dtype=np.uint8)
                association_count = len(relations["association/shape"])
                spatial_examples += len(spatial)
                temporal_examples += len(temporal)
                associations += association_count
                if len(spatial):
                    spatial_counts += spatial.sum(axis=0, dtype=np.int64)
                if len(temporal):
                    temporal_counts += temporal.sum(axis=0, dtype=np.int64)
            sequence_rows[sequence] = {
                "objects": int(len(labels)),
                "full_multimodal_objects": int(eligible.sum()),
                "spatial_examples": int(len(spatial)),
                "temporal_examples": int(len(temporal)),
                "association_sample_pairs": int(association_count),
            }
        unsupported_spatial = [
            name for name, count in zip(SPATIAL_PREDICATES, spatial_counts) if count == 0
        ]
        unsupported_temporal = [
            name for name, count in zip(TEMPORAL_PREDICATES, temporal_counts) if count == 0
        ]
        if not spatial_examples or not temporal_examples or not associations:
            errors.append(f"{split_name} contains an empty full-multimodal task")
        if unsupported_spatial or unsupported_temporal:
            errors.append(
                f"{split_name} has unsupported predicates: "
                f"spatial={unsupported_spatial}, temporal={unsupported_temporal}"
            )
        result["splits"][split_name] = {
            "sequences": sequences,
            "objects": total_objects,
            "full_multimodal_objects": full_objects,
            "full_multimodal_object_percent": (
                100.0 * full_objects / total_objects if total_objects else 0.0
            ),
            "object_class_counts": dict(zip(OBJECT_CLASSES, object_counts.tolist())),
            "spatial_examples": spatial_examples,
            "spatial_positive_counts": dict(zip(SPATIAL_PREDICATES, spatial_counts.tolist())),
            "temporal_examples": temporal_examples,
            "temporal_positive_counts": dict(zip(TEMPORAL_PREDICATES, temporal_counts.tolist())),
            "association_sample_pairs": associations,
            "sequence_details": sequence_rows,
        }
    result["passed"] = not errors
    result["errors"] = errors
    atomic_json(args.output.resolve(), result)
    if errors:
        raise RuntimeError("Full-multimodal audit failed: " + "; ".join(errors))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict KITTI-360 multimodal shards.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2))


if __name__ == "__main__":
    main()
