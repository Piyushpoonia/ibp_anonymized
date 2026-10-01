from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .prepare_features import stream_json_array
from .protocol import SPATIAL_PREDICATES, TEMPORAL_PREDICATES


SCHEMA_VERSION = "IBP-K360-relation-index-v1.2.0"


def is_full_multimodal(row: int, modality_mask: np.ndarray) -> bool:
    return row >= 0 and bool(np.asarray(modality_mask[row], dtype=np.bool_).all())


def joint_multimodal_mask(
    source_rows: list[int], target_rows: list[int], modality_mask: np.ndarray
) -> list[int]:
    return [
        int(is_full_multimodal(source, modality_mask) and is_full_multimodal(target, modality_mask))
        for source, target in zip(source_rows, target_rows)
    ]


def feature_lookup(metadata_path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                result[str(row["annotation_token"])] = int(row["feature_row"])
    return result


def spatial_geometry(row: dict[str, Any]) -> np.ndarray:
    volume_ratio = max(float(row["volume_ratio_subject_object"]), 1e-8)
    height_ratio = max(float(row["height_ratio_subject_object"]), 1e-8)
    return np.asarray(
        [
            *row["delta_ego_xyz"],
            row["horizontal_center_distance_m"],
            row["oriented_box_distance_m"],
            row["bev_intersection_m2"],
            row["bev_iou"],
            row["vertical_overlap_m"],
            row["image_overlap_over_smaller"],
            math.log(volume_ratio),
            math.log(height_ratio),
            row["lateral_margin_m"],
            row["longitudinal_margin_m"],
            int(row["subject_mask_pixel_count"] > 0),
            int(row["object_mask_pixel_count"] > 0),
            int(row["occlusion_mask_supported"]),
        ],
        dtype=np.float32,
    )


def temporal_geometry(row: dict[str, Any]) -> np.ndarray:
    distances = list(row["relative_distances_m"])
    distances = (distances + [0.0] * 5)[:5]
    observed_path = row.get("observed_path_distance_m")
    return np.asarray(
        [
            *distances,
            row["distance_change_m"],
            row["decreasing_step_fraction"],
            row["increasing_step_fraction"],
            row["subject_speed_mps"],
            row["object_speed_mps"],
            *row["subject_velocity_world_mps"],
            *row["object_velocity_world_mps"],
            row["subject_radial_velocity_toward_object_mps"],
            row["object_radial_velocity_toward_subject_mps"],
            row["motion_direction_cosine"],
            row["middle_distance_m"],
            row["subject_visible_steps"] / 5.0,
            row["object_visible_steps"] / 5.0,
            row["joint_visible_steps"] / 5.0,
            0.0 if observed_path is None else observed_path,
        ],
        dtype=np.float32,
    )


class BufferedPairs:
    def __init__(self, group: h5py.Group, geometry_dim: int, labels: int, tracklets: bool):
        self.group = group
        self.geometry_dim = geometry_dim
        self.labels = labels
        self.tracklets = tracklets
        prefix_shape = (0, 5) if tracklets else (0,)
        max_prefix = (None, 5) if tracklets else (None,)
        self.source = group.create_dataset(
            "source_rows", shape=prefix_shape, maxshape=max_prefix, dtype="int64", chunks=True
        )
        self.target = group.create_dataset(
            "target_rows", shape=prefix_shape, maxshape=max_prefix, dtype="int64", chunks=True
        )
        self.source_mask = None
        self.target_mask = None
        if tracklets:
            self.source_mask = group.create_dataset(
                "source_mask", shape=(0, 5), maxshape=(None, 5), dtype="uint8", chunks=True
            )
            self.target_mask = group.create_dataset(
                "target_mask", shape=(0, 5), maxshape=(None, 5), dtype="uint8", chunks=True
            )
        self.geometry = group.create_dataset(
            "geometry",
            shape=(0, geometry_dim),
            maxshape=(None, geometry_dim),
            dtype="float32",
            chunks=True,
        )
        self.targets = group.create_dataset(
            "targets", shape=(0, labels), maxshape=(None, labels), dtype="uint8", chunks=True
        )
        self.relation_ids = group.create_dataset(
            "relation_ids", shape=(0,), maxshape=(None,), dtype="S32", chunks=True
        )
        self.source_annotation_tokens = group.create_dataset(
            "source_annotation_tokens",
            shape=prefix_shape,
            maxshape=max_prefix,
            dtype="S32",
            chunks=True,
        )
        self.target_annotation_tokens = group.create_dataset(
            "target_annotation_tokens",
            shape=prefix_shape,
            maxshape=max_prefix,
            dtype="S32",
            chunks=True,
        )
        self.raw_frame_indices = group.create_dataset(
            "raw_frame_indices",
            shape=prefix_shape,
            maxshape=max_prefix,
            dtype="int64",
            chunks=True,
        )
        self.group_ids = group.create_dataset(
            "group_ids", shape=(0,), maxshape=(None,), dtype="S64", chunks=True
        )
        self.buffer: list[tuple[Any, ...]] = []

    def append(self, *values: Any) -> None:
        self.buffer.append(values)
        if len(self.buffer) >= 8192:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        start = len(self.targets)
        end = start + len(self.buffer)
        datasets = [self.source, self.target]
        if self.tracklets:
            datasets.extend([self.source_mask, self.target_mask])
        datasets.extend(
            [
                self.geometry,
                self.targets,
                self.relation_ids,
                self.source_annotation_tokens,
                self.target_annotation_tokens,
                self.raw_frame_indices,
                self.group_ids,
            ]
        )
        for dataset in datasets:
            dataset.resize(end, axis=0)
        columns = list(zip(*self.buffer))
        self.source[start:end] = np.asarray(columns[0], dtype=np.int64)
        self.target[start:end] = np.asarray(columns[1], dtype=np.int64)
        offset = 2
        if self.tracklets:
            self.source_mask[start:end] = np.asarray(columns[offset], dtype=np.uint8)
            self.target_mask[start:end] = np.asarray(columns[offset + 1], dtype=np.uint8)
            offset += 2
        self.geometry[start:end] = np.asarray(columns[offset], dtype=np.float32)
        self.targets[start:end] = np.asarray(columns[offset + 1], dtype=np.uint8)
        self.relation_ids[start:end] = np.asarray(columns[offset + 2], dtype="S32")
        self.source_annotation_tokens[start:end] = np.asarray(
            columns[offset + 3], dtype="S32"
        )
        self.target_annotation_tokens[start:end] = np.asarray(
            columns[offset + 4], dtype="S32"
        )
        self.raw_frame_indices[start:end] = np.asarray(columns[offset + 5], dtype=np.int64)
        self.group_ids[start:end] = np.asarray(columns[offset + 6], dtype="S64")
        self.buffer.clear()


def build_spatial(
    path: Path,
    lookup: dict[str, int],
    writer: BufferedPairs,
    modality_mask: np.ndarray,
    full_multimodal_only: bool,
) -> dict[str, int]:
    counts = {"seen": 0, "kept": 0, "missing_features": 0, "non_multimodal_pairs": 0}
    for row in stream_json_array(path):
        counts["seen"] += 1
        source = lookup.get(str(row["subject_annotation_token"]))
        target = lookup.get(str(row["object_annotation_token"]))
        if source is None or target is None:
            counts["missing_features"] += 1
            continue
        if full_multimodal_only and not (
            is_full_multimodal(source, modality_mask)
            and is_full_multimodal(target, modality_mask)
        ):
            counts["non_multimodal_pairs"] += 1
            continue
        labels = [int(bool(row[name])) for name in SPATIAL_PREDICATES]
        writer.append(
            source,
            target,
            spatial_geometry(row),
            labels,
            str(row["token"]),
            str(row["subject_annotation_token"]),
            str(row["object_annotation_token"]),
            int(row["raw_frame_index"]),
            str(row["sample_token"]),
        )
        counts["kept"] += 1
    writer.flush()
    return counts


def tracklet_rows(tokens: list[str], lookup: dict[str, int]) -> tuple[list[int], list[int]]:
    rows = [lookup.get(str(token), -1) for token in tokens]
    mask = [int(row >= 0) for row in rows]
    return rows, mask


def build_temporal(
    path: Path,
    lookup: dict[str, int],
    writer: BufferedPairs,
    modality_mask: np.ndarray,
    full_multimodal_only: bool,
) -> dict[str, int]:
    counts = {
        "seen": 0,
        "kept": 0,
        "insufficient_tracklets": 0,
        "non_multimodal_tracklets": 0,
        "exploratory_ignored": 0,
    }
    for row in stream_json_array(path):
        counts["seen"] += 1
        source, source_mask = tracklet_rows(row["subject_annotation_tokens"], lookup)
        target, target_mask = tracklet_rows(row["object_annotation_tokens"], lookup)
        if sum(source_mask) < 3 or sum(target_mask) < 3:
            counts["insufficient_tracklets"] += 1
            continue
        if full_multimodal_only:
            joint_mask = joint_multimodal_mask(source, target, modality_mask)
            middle = len(joint_mask) // 2
            if sum(joint_mask) < 3 or not joint_mask[middle]:
                counts["non_multimodal_tracklets"] += 1
                continue
            source_mask = joint_mask
            target_mask = joint_mask
        labels = [int(bool(row[name])) for name in TEMPORAL_PREDICATES]
        counts["exploratory_ignored"] += int(bool(row.get("crossing_path", False)))
        writer.append(
            source,
            target,
            source_mask,
            target_mask,
            temporal_geometry(row),
            labels,
            str(row["token"]),
            [str(token) for token in row["subject_annotation_tokens"]],
            [str(token) for token in row["object_annotation_tokens"]],
            [int(value) for value in row["raw_frame_indices"]],
            f'{row["scene_token"]}:{int(row["start_sample_index"])}',
        )
        counts["kept"] += 1
    writer.flush()
    return counts


def association_geometry(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    feature_geometry: dict[str, np.ndarray],
) -> np.ndarray:
    source_center = feature_geometry["center_world"][source_rows]
    target_center = feature_geometry["center_world"][target_rows]
    source_size = np.maximum(feature_geometry["box_size"][source_rows], 1e-4)
    target_size = np.maximum(feature_geometry["box_size"][target_rows], 1e-4)
    source_mask = feature_geometry["modality_mask"][source_rows]
    target_mask = feature_geometry["modality_mask"][target_rows]
    delta = target_center[None, :, :] - source_center[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    direction = delta / np.maximum(distance, 1e-6)
    size_difference = target_size[None, :, :] - source_size[:, None, :]
    log_size_ratio = np.log(target_size[None, :, :] / source_size[:, None, :])
    modality_overlap = source_mask[:, None, :] * target_mask[None, :, :]
    return np.concatenate(
        [delta, distance, direction, size_difference, log_size_ratio, modality_overlap], axis=-1
    ).astype(np.float32)


def build_associations(
    path: Path,
    lookup: dict[str, int],
    feature_geometry: dict[str, np.ndarray],
    group: h5py.Group,
    full_multimodal_only: bool,
) -> dict[str, int]:
    frame = pd.read_parquet(path)
    frame = frame[frame["supervised_eligible"].astype(bool)]
    variable_int = h5py.vlen_dtype(np.dtype("int64"))
    variable_float = h5py.vlen_dtype(np.dtype("float32"))
    variable_byte = h5py.vlen_dtype(np.dtype("uint8"))
    source_rows_ds = group.create_dataset("source_rows", (0,), maxshape=(None,), dtype=variable_int)
    target_rows_ds = group.create_dataset("target_rows", (0,), maxshape=(None,), dtype=variable_int)
    geometry_ds = group.create_dataset("geometry", (0,), maxshape=(None,), dtype=variable_float)
    candidate_ds = group.create_dataset("candidate_mask", (0,), maxshape=(None,), dtype=variable_byte)
    source_targets_ds = group.create_dataset("source_targets", (0,), maxshape=(None,), dtype=variable_int)
    target_targets_ds = group.create_dataset("target_targets", (0,), maxshape=(None,), dtype=variable_int)
    shapes_ds = group.create_dataset("shape", (0, 2), maxshape=(None, 2), dtype="int32")
    source_sample_ds = group.create_dataset(
        "source_sample_tokens", (0,), maxshape=(None,), dtype="S32"
    )
    target_sample_ds = group.create_dataset(
        "target_sample_tokens", (0,), maxshape=(None,), dtype="S32"
    )

    counts = {
        "sample_pairs_seen": 0,
        "sample_pairs_kept": 0,
        "missing_features": 0,
        "non_multimodal_source_objects": 0,
        "non_multimodal_target_objects": 0,
    }
    grouped = frame.groupby(["source_sample_token", "target_sample_token"], sort=False)
    for (source_sample_token, target_sample_token), candidates in grouped:
        counts["sample_pairs_seen"] += 1
        object_pairs = candidates[candidates["candidate_type"] == "object_pair"]
        source_tokens = list(dict.fromkeys(candidates["source_annotation_token"].astype(str)))
        target_tokens = list(dict.fromkeys(candidates["target_annotation_token"].astype(str)))
        source_tokens = [token for token in source_tokens if token]
        target_tokens = [token for token in target_tokens if token]
        if any(token not in lookup for token in source_tokens + target_tokens):
            counts["missing_features"] += 1
            continue
        if full_multimodal_only:
            source_before = len(source_tokens)
            target_before = len(target_tokens)
            source_tokens = [
                token
                for token in source_tokens
                if is_full_multimodal(lookup[token], feature_geometry["modality_mask"])
            ]
            target_tokens = [
                token
                for token in target_tokens
                if is_full_multimodal(lookup[token], feature_geometry["modality_mask"])
            ]
            counts["non_multimodal_source_objects"] += source_before - len(source_tokens)
            counts["non_multimodal_target_objects"] += target_before - len(target_tokens)
        source_rows = np.asarray([lookup[token] for token in source_tokens], dtype=np.int64)
        target_rows = np.asarray([lookup[token] for token in target_tokens], dtype=np.int64)
        source_index = {token: index for index, token in enumerate(source_tokens)}
        target_index = {token: index for index, token in enumerate(target_tokens)}
        ns, nt = len(source_tokens), len(target_tokens)
        if ns == 0 or nt == 0:
            continue
        candidate_mask = np.zeros((ns, nt), dtype=np.uint8)
        source_targets = np.full(ns, nt, dtype=np.int64)
        target_targets = np.full(nt, ns, dtype=np.int64)
        for row in object_pairs.itertuples(index=False):
            source_token = str(row.source_annotation_token)
            target_token = str(row.target_annotation_token)
            if source_token not in source_index or target_token not in target_index:
                continue
            source = source_index[source_token]
            target = target_index[target_token]
            candidate_mask[source, target] = 1
            if bool(row.forward_target):
                source_targets[source] = target
            if bool(row.reverse_target):
                target_targets[target] = source

        index = len(shapes_ds)
        for dataset in (
            source_rows_ds,
            target_rows_ds,
            geometry_ds,
            candidate_ds,
            source_targets_ds,
            target_targets_ds,
            shapes_ds,
            source_sample_ds,
            target_sample_ds,
        ):
            dataset.resize(index + 1, axis=0)
        source_rows_ds[index] = source_rows
        target_rows_ds[index] = target_rows
        geometry_ds[index] = association_geometry(source_rows, target_rows, feature_geometry).reshape(-1)
        candidate_ds[index] = candidate_mask.reshape(-1)
        source_targets_ds[index] = source_targets
        target_targets_ds[index] = target_targets
        shapes_ds[index] = (ns, nt)
        source_sample_ds[index] = str(source_sample_token).encode("ascii")
        target_sample_ds[index] = str(target_sample_token).encode("ascii")
        counts["sample_pairs_kept"] += 1
    return counts


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Relation index already exists: {output}; use --force to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    lookup = feature_lookup(args.feature_metadata.resolve())
    with h5py.File(args.feature_file.resolve(), "r") as features, h5py.File(temporary, "w") as target:
        target.attrs["schema_version"] = SCHEMA_VERSION
        target.attrs["full_multimodal_only"] = bool(args.full_multimodal_only)
        target.attrs["sequence"] = args.feature_file.stem
        spatial_writer = BufferedPairs(target.create_group("spatial"), 16, 5, False)
        temporal_writer = BufferedPairs(target.create_group("temporal"), 24, 3, True)
        feature_geometry = {
            key: np.asarray(features[key])
            for key in ("center_world", "box_size", "modality_mask")
        }
        spatial_counts = build_spatial(
            args.spatial_relations.resolve(),
            lookup,
            spatial_writer,
            feature_geometry["modality_mask"],
            args.full_multimodal_only,
        )
        temporal_counts = build_temporal(
            args.temporal_relations.resolve(),
            lookup,
            temporal_writer,
            feature_geometry["modality_mask"],
            args.full_multimodal_only,
        )
        association_counts = build_associations(
            args.association_candidates.resolve(),
            lookup,
            feature_geometry,
            target.create_group("association"),
            args.full_multimodal_only,
        )
    temporary.replace(output)
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_rows": len(lookup),
        "full_multimodal_only": bool(args.full_multimodal_only),
        "full_multimodal_definition": (
            "Spatial: both objects have RGB+LiDAR+text. Temporal: both objects jointly have "
            "all modalities in at least three of five steps, including the middle step."
        ),
        "spatial": spatial_counts,
        "temporal": temporal_counts,
        "association": association_counts,
        "spatial_predicates": list(SPATIAL_PREDICATES),
        "temporal_predicates": list(TEMPORAL_PREDICATES),
        "identity_privacy": "Persistent identity was used only to construct association targets.",
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact Stage-2 indexes for KITTI-360 IBP.")
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--spatial-relations", type=Path, required=True)
    parser.add_argument("--temporal-relations", type=Path, required=True)
    parser.add_argument("--association-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--full-multimodal-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_index(parse_args()), indent=2))


if __name__ == "__main__":
    main()
