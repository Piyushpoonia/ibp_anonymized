#!/usr/bin/env python3
"""Build KITTI-360 object observations and provisional spatial predicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint


SCHEMA_VERSION = "IBP-K360-predicates-v3.0.0-canonical-candidate"
PREDICATES = [
    "left_of",
    "in_front_of",
    "near",
    "overlapping",
    "occluding",
]
SYMMETRIC_PREDICATES = ("near", "overlapping")
ASYMMETRIC_PREDICATES = ("left_of", "in_front_of", "occluding")
CANONICAL_INVERSE_ENCODING = {
    "right_of(A,B)": "left_of(B,A)",
    "behind(A,B)": "in_front_of(B,A)",
    "occluded_by(A,B)": "occluding(B,A)",
}


def stable_token(*parts: object) -> str:
    source = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(source, digest_size=16).hexdigest()


def matrix_from_flat(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is pd.NA:
        return None
    return value


def atomic_json_records(records: object, path: Path) -> None:
    """Write a standard JSON array incrementally without retaining a JSON string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("[\n")
        first = True
        for record in records:
            if not first:
                stream.write(",\n")
            json.dump(json_safe(record), stream, ensure_ascii=True, separators=(",", ":"))
            first = False
        stream.write("\n]\n")
    os.replace(temporary, path)


def read_perspective(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = np.asarray(
                [float(item) for item in raw.split()], dtype=np.float64
            )
        except ValueError:
            continue
    return values


def xml_matrix(node: ET.Element, tag: str) -> np.ndarray:
    matrix = node.find(tag)
    if matrix is None:
        raise ValueError(f"Missing {tag}")
    rows = int(matrix.findtext("rows", "0"))
    cols = int(matrix.findtext("cols", "0"))
    raw = matrix.findtext("data", "")
    values = np.fromstring(raw, sep=" ", dtype=np.float64)
    if values.size != rows * cols:
        raise ValueError(f"Invalid {tag}: expected {rows * cols}, found {values.size}")
    return values.reshape(rows, cols)


@dataclass(frozen=True)
class BoxRecord:
    annotation_index: int
    raw_label: str
    semantic_id: int
    instance_id: int
    dynamic: bool
    timestamp: int
    start_frame: int
    end_frame: int
    transform_world: np.ndarray
    vertices_local: np.ndarray
    source_xml: str

    @property
    def identity(self) -> tuple[int, int]:
        return self.semantic_id, self.instance_id


def load_boxes(path: Path, dataset_root: Path) -> list[BoxRecord]:
    root = ET.parse(path).getroot()
    records: list[BoxRecord] = []
    source = str(path.relative_to(dataset_root)).replace("\\", "/")
    for child in root:
        if child.find("transform") is None or child.find("vertices") is None:
            continue
        try:
            timestamp = int(child.findtext("timestamp", "-1"))
            records.append(
                BoxRecord(
                    annotation_index=int(child.findtext("index", "-1")),
                    raw_label=child.findtext("label", "unknown").strip(),
                    semantic_id=int(child.findtext("semanticId", "-1")),
                    instance_id=int(child.findtext("instanceId", "-1")),
                    dynamic=bool(int(child.findtext("dynamic", "0"))),
                    timestamp=timestamp,
                    start_frame=int(child.findtext("start_frame", str(timestamp))),
                    end_frame=int(child.findtext("end_frame", str(timestamp))),
                    transform_world=xml_matrix(child, "transform"),
                    vertices_local=xml_matrix(child, "vertices"),
                    source_xml=source,
                )
            )
        except (TypeError, ValueError):
            continue
    return records


def active_boxes(
    frame: int,
    static_boxes: list[BoxRecord],
    dynamic_by_frame: dict[int, list[BoxRecord]],
) -> list[BoxRecord]:
    selected: dict[tuple[int, int], BoxRecord] = {}
    for box in static_boxes:
        if box.start_frame <= frame <= box.end_frame:
            selected[box.identity] = box
    for box in dynamic_by_frame.get(frame, []):
        selected[box.identity] = box
    return list(selected.values())


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (transform @ homogeneous.T).T[:, :3]


def project_box(
    corners_world: np.ndarray,
    transform_world_camera: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[float] | None, float, float]:
    camera_from_world = np.linalg.inv(transform_world_camera)
    camera = transform_points(camera_from_world, corners_world)
    rectified = (rectification @ camera.T).T
    valid = rectified[:, 2] > 0.10
    if valid.sum() < 2:
        return None, 0.0, math.nan
    points = rectified[valid]
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (projection @ homogeneous.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    x1 = float(np.clip(uv[:, 0].min(), 0, width - 1))
    y1 = float(np.clip(uv[:, 1].min(), 0, height - 1))
    x2 = float(np.clip(uv[:, 0].max(), 0, width - 1))
    y2 = float(np.clip(uv[:, 1].max(), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, float(np.median(points[:, 2]))
    return [x1, y1, x2, y2], (x2 - x1) * (y2 - y1), float(
        np.median(points[:, 2])
    )


def count_lidar_points(
    tree: cKDTree | None,
    lidar_xyz: np.ndarray | None,
    center_ego: np.ndarray,
    corners_ego: np.ndarray,
    transform_world_object: np.ndarray,
    transform_world_velo: np.ndarray,
    vertices_local: np.ndarray,
) -> int:
    if tree is None or lidar_xyz is None:
        return 0
    radius = float(np.linalg.norm(corners_ego - center_ego, axis=1).max() + 0.20)
    indices = tree.query_ball_point(center_ego, radius)
    if not indices:
        return 0
    candidates = lidar_xyz[np.asarray(indices, dtype=np.int64)]
    local_from_velo = np.linalg.inv(transform_world_object) @ transform_world_velo
    local = transform_points(local_from_velo, candidates)
    lower = vertices_local.min(axis=0) - 1e-3
    upper = vertices_local.max(axis=0) + 1e-3
    inside = np.logical_and(local >= lower, local <= upper).all(axis=1)
    return int(inside.sum())


def mask_counts(path: Path | None) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    image = np.asarray(Image.open(path))
    values, counts = np.unique(image, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def frame_data_lookup(sample_data: pd.DataFrame) -> dict[tuple[str, str], str]:
    key = sample_data[sample_data["is_key_frame"].astype(bool)]
    result: dict[tuple[str, str], str] = {}
    for row in key.itertuples(index=False):
        result[(str(row.sample_token), str(row.channel))] = str(row.filename)
    return result


def uniform_subset(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or maximum >= len(frame):
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, maximum, dtype=np.int64)
    return frame.iloc[np.unique(indices)].copy()


def bbox_overlap_over_smaller(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = min(area_first, area_second)
    return intersection / denominator if denominator > 0 else 0.0


def relation_row(
    sample_token: str,
    sequence: str,
    subject: dict[str, Any],
    target: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    center_a = np.asarray(subject["center_ego"], dtype=np.float64)
    center_b = np.asarray(target["center_ego"], dtype=np.float64)
    delta = center_a - center_b
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    if horizontal_distance > args.candidate_distance:
        return None

    corners_a = np.asarray(subject["corners_ego"], dtype=np.float64).reshape(-1, 3)
    corners_b = np.asarray(target["corners_ego"], dtype=np.float64).reshape(-1, 3)
    polygon_a = MultiPoint(corners_a[:, :2]).convex_hull
    polygon_b = MultiPoint(corners_b[:, :2]).convex_hull
    if polygon_a.is_empty or polygon_b.is_empty or polygon_a.area <= 0 or polygon_b.area <= 0:
        return None

    intersection_area = float(polygon_a.intersection(polygon_b).area)
    union_area = float(polygon_a.union(polygon_b).area)
    bev_iou = intersection_area / union_area if union_area > 0 else 0.0
    box_distance = float(polygon_a.distance(polygon_b))
    z_min_a, z_max_a = float(corners_a[:, 2].min()), float(corners_a[:, 2].max())
    z_min_b, z_max_b = float(corners_b[:, 2].min()), float(corners_b[:, 2].max())
    height_a = max(z_max_a - z_min_a, 1e-6)
    height_b = max(z_max_b - z_min_b, 1e-6)
    vertical_overlap = max(0.0, min(z_max_a, z_max_b) - max(z_min_a, z_min_b))
    extent_y_a = float(corners_a[:, 1].max() - corners_a[:, 1].min())
    extent_y_b = float(corners_b[:, 1].max() - corners_b[:, 1].min())
    extent_x_a = float(corners_a[:, 0].max() - corners_a[:, 0].min())
    extent_x_b = float(corners_b[:, 0].max() - corners_b[:, 0].min())
    lateral_margin = max(args.lateral_min_margin, 0.25 * (extent_y_a + extent_y_b))
    longitudinal_margin = max(
        args.longitudinal_min_margin,
        0.25 * (extent_x_a + extent_x_b),
    )

    volume_a = max(float(subject["box_volume_m3"]), 1e-6)
    volume_b = max(float(target["box_volume_m3"]), 1e-6)
    bbox_a = subject.get("projected_bbox_xyxy")
    bbox_b = target.get("projected_bbox_xyxy")
    image_overlap = (
        bbox_overlap_over_smaller(bbox_a, bbox_b)
        if isinstance(bbox_a, list) and isinstance(bbox_b, list)
        else 0.0
    )
    depth_a = float(subject.get("camera_depth_m", math.nan))
    depth_b = float(target.get("camera_depth_m", math.nan))
    local_pair = horizontal_distance <= args.relation_distance
    occlusion_mask_supported = bool(
        subject.get("mask_available", False)
        and target.get("mask_available", False)
        and int(subject.get("mask_pixel_count", 0)) >= args.minimum_mask_pixels
        and int(target.get("mask_pixel_count", 0)) >= args.minimum_mask_pixels
    )

    labels = {
        "left_of": bool(local_pair and delta[1] > lateral_margin),
        "in_front_of": bool(local_pair and delta[0] > longitudinal_margin),
        "near": bool(box_distance <= args.near_distance),
        "overlapping": bool(
            bev_iou >= args.overlap_bev_iou
            and vertical_overlap >= args.overlap_vertical
        ),
        "occluding": bool(
            occlusion_mask_supported
            and image_overlap >= args.occlusion_overlap
            and np.isfinite(depth_a)
            and np.isfinite(depth_b)
            and depth_a + args.occlusion_depth_margin < depth_b
        ),
    }
    vector = [int(labels[name]) for name in PREDICATES]
    positives = [name for name in PREDICATES if labels[name]]
    row: dict[str, Any] = {
        "token": stable_token(
            "relation-v3-canonical", sample_token, subject["token"], target["token"]
        ),
        "sample_token": sample_token,
        "sequence": sequence,
        "raw_frame_index": int(subject["raw_frame_index"]),
        "rgb_left_filename": str(subject["cam0_filename"]),
        "subject_annotation_token": subject["token"],
        "object_annotation_token": target["token"],
        "subject_instance_token": subject["instance_token"],
        "object_instance_token": target["instance_token"],
        "subject_label": subject["raw_label"],
        "object_label": target["raw_label"],
        "subject_projected_bbox_xyxy": bbox_a,
        "object_projected_bbox_xyxy": bbox_b,
        "delta_ego_xyz": delta.tolist(),
        "horizontal_center_distance_m": horizontal_distance,
        "oriented_box_distance_m": box_distance,
        "bev_intersection_m2": intersection_area,
        "bev_iou": bev_iou,
        "vertical_overlap_m": vertical_overlap,
        "image_overlap_over_smaller": image_overlap,
        "subject_mask_pixel_count": int(subject.get("mask_pixel_count", 0)),
        "object_mask_pixel_count": int(target.get("mask_pixel_count", 0)),
        "occlusion_mask_supported": occlusion_mask_supported,
        "lateral_margin_m": lateral_margin,
        "longitudinal_margin_m": longitudinal_margin,
        "volume_ratio_subject_object": volume_a / volume_b,
        "height_ratio_subject_object": height_a / height_b,
        "predicate_vector": vector,
        "positive_predicates": positives,
        "no_relation": not any(vector),
        "supervised_eligible": bool(
            subject["supervised_eligible"] and target["supervised_eligible"]
        ),
        "schema_version": SCHEMA_VERSION,
    }
    row.update(labels)
    return row


def write_relation_json(
    annotations: pd.DataFrame,
    path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, int], int, int, list[dict[str, str]]]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    predicate_counts = Counter({name: 0 for name in PREDICATES})
    relation_count = 0
    no_relation_count = 0
    canonical_violations: list[dict[str, str]] = []
    first = True
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("[\n")
        for sample_token, group in annotations.groupby("sample_token", sort=False):
            records = group.to_dict("records")
            sample_rows: dict[tuple[str, str], dict[str, Any]] = {}
            for subject in records:
                for target in records:
                    if subject["token"] == target["token"]:
                        continue
                    row = relation_row(
                        str(sample_token), str(subject["sequence"]), subject, target, args
                    )
                    if row is None:
                        continue
                    key = (row["subject_annotation_token"], row["object_annotation_token"])
                    sample_rows[key] = row
                    relation_count += 1
                    no_relation_count += int(row["no_relation"])
                    for name in row["positive_predicates"]:
                        predicate_counts[name] += 1
                    if not first:
                        stream.write(",\n")
                    json.dump(
                        json_safe(row),
                        stream,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    first = False

            for (subject_token, object_token), row in sample_rows.items():
                reverse = sample_rows.get((object_token, subject_token))
                if reverse is None:
                    canonical_violations.append(
                        {
                            "sample_token": str(sample_token),
                            "error": "missing_reverse_pair",
                        }
                    )
                    continue
                for name in SYMMETRIC_PREDICATES:
                    if bool(row[name]) != bool(reverse[name]):
                        canonical_violations.append(
                            {
                                "sample_token": str(sample_token),
                                "subject": subject_token,
                                "object": object_token,
                                "predicate": name,
                                "error": "symmetric_predicate_mismatch",
                            }
                        )
                        break
                for name in ASYMMETRIC_PREDICATES:
                    if bool(row[name]) and bool(reverse[name]):
                        canonical_violations.append(
                            {
                                "sample_token": str(sample_token),
                                "subject": subject_token,
                                "object": object_token,
                                "predicate": name,
                                "error": "asymmetric_predicate_true_in_both_directions",
                            }
                        )
                        break
                if len(canonical_violations) >= 100:
                    break
        stream.write("\n]\n")
    os.replace(temporary, path)
    return dict(predicate_counts), relation_count, no_relation_count, canonical_violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-lidar-count", action="store_true")
    parser.add_argument("--minimum-rgb-size", type=float, default=20.0)
    parser.add_argument("--minimum-mask-pixels", type=int, default=20)
    parser.add_argument("--minimum-lidar-points", type=int, default=10)
    parser.add_argument("--candidate-distance", type=float, default=30.0)
    parser.add_argument("--relation-distance", type=float, default=20.0)
    parser.add_argument("--near-distance", type=float, default=3.0)
    parser.add_argument("--lateral-min-margin", type=float, default=0.5)
    parser.add_argument("--longitudinal-min-margin", type=float, default=1.0)
    parser.add_argument("--overlap-bev-iou", type=float, default=0.01)
    parser.add_argument("--overlap-vertical", type=float, default=0.10)
    parser.add_argument("--occlusion-overlap", type=float, default=0.25)
    parser.add_argument("--occlusion-depth-margin", type=float, default=0.50)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    timeline_root = args.timeline_root.resolve()
    output_root = args.output_root.resolve()
    tables = timeline_root / "tables"
    if not tables.exists():
        raise FileNotFoundError(f"Timeline tables not found: {tables}")
    result_file = output_root / "predicate_build_report.json"
    if result_file.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {result_file}. Pass --force to replace.")
    output_root.mkdir(parents=True, exist_ok=True)

    logs = pd.read_parquet(tables / "log.parquet")
    if len(logs) != 1:
        raise ValueError("Predicate partitions must contain exactly one KITTI-360 sequence")
    sequence = str(logs.iloc[0]["name"])
    log_token = str(logs.iloc[0]["token"])
    samples = pd.read_parquet(tables / "sample.parquet").sort_values(
        ["timestamp_ns", "raw_frame_index"]
    )
    samples = uniform_subset(samples, args.max_samples)
    sample_data = pd.read_parquet(tables / "sample_data.parquet")
    poses = pd.read_parquet(tables / "ego_pose.parquet")
    pose_by_frame = {
        int(row.raw_frame_index): row
        for row in poses.itertuples(index=False)
        if int(row.raw_frame_index) in set(samples["raw_frame_index"].astype(int))
    }
    data_lookup = frame_data_lookup(sample_data)

    box_path = dataset_root / f"data_3d_bboxes/train/{sequence}.xml"
    if not box_path.exists():
        raise FileNotFoundError(f"3D bounding-box XML not found: {box_path}")
    boxes = load_boxes(box_path, dataset_root)
    static_boxes = [box for box in boxes if not box.dynamic]
    dynamic_by_frame: dict[int, list[BoxRecord]] = defaultdict(list)
    for box in boxes:
        if box.dynamic:
            dynamic_by_frame[box.timestamp].append(box)

    perspective = read_perspective(dataset_root / "calibration/perspective.txt")
    rectification = perspective["R_rect_00"].reshape(3, 3)
    projection = perspective["P_rect_00"].reshape(3, 4)
    width, height = (int(item) for item in perspective["S_rect_00"])

    annotation_rows: list[dict[str, Any]] = []
    skipped_singular = 0
    skipped_unobservable = 0
    skipped_missing_pose = 0
    for progress, sample in enumerate(samples.itertuples(index=False), start=1):
        frame = int(sample.raw_frame_index)
        pose = pose_by_frame.get(frame)
        if pose is None:
            skipped_missing_pose += 1
            continue
        transform_world_velo = matrix_from_flat(pose.T_world_velo)
        transform_world_camera = matrix_from_flat(pose.T_world_camera0)
        velo_from_world = np.linalg.inv(transform_world_velo)

        cam0_relative = data_lookup.get((str(sample.token), "CAM0"))
        lidar_relative = data_lookup.get((str(sample.token), "LIDAR_TOP"))
        mask_relative = data_lookup.get((str(sample.token), "INSTANCE_CAM0"))
        cam0_path = dataset_root / cam0_relative if cam0_relative else None
        lidar_path = dataset_root / lidar_relative if lidar_relative else None
        mask_path = dataset_root / mask_relative if mask_relative else None
        per_mask_counts = mask_counts(mask_path)

        lidar_xyz: np.ndarray | None = None
        tree: cKDTree | None = None
        if not args.skip_lidar_count and lidar_path is not None and lidar_path.exists():
            raw_lidar = np.fromfile(lidar_path, dtype=np.float32)
            if raw_lidar.size % 4 != 0:
                raise ValueError(f"Invalid Velodyne file: {lidar_path}")
            lidar_xyz = raw_lidar.reshape(-1, 4)[:, :3].astype(np.float64, copy=False)
            tree = cKDTree(lidar_xyz)

        for box in active_boxes(frame, static_boxes, dynamic_by_frame):
            try:
                corners_world = transform_points(box.transform_world, box.vertices_local)
                center_world = box.transform_world[:3, 3]
                corners_ego = transform_points(velo_from_world, corners_world)
                center_ego = transform_points(velo_from_world, center_world[None, :])[0]
                projected_bbox, projected_area, camera_depth = project_box(
                    corners_world,
                    transform_world_camera,
                    rectification,
                    projection,
                    width,
                    height,
                )
                lidar_count = count_lidar_points(
                    tree,
                    lidar_xyz,
                    center_ego,
                    corners_ego,
                    box.transform_world,
                    transform_world_velo,
                    box.vertices_local,
                )
            except np.linalg.LinAlgError:
                skipped_singular += 1
                continue

            combined_instance_id = box.semantic_id * 1000 + box.instance_id
            mask_pixel_count = per_mask_counts.get(combined_instance_id, 0)
            mask_available = mask_path is not None and mask_path.exists()
            bbox_width = (
                projected_bbox[2] - projected_bbox[0] if projected_bbox is not None else 0.0
            )
            bbox_height = (
                projected_bbox[3] - projected_bbox[1] if projected_bbox is not None else 0.0
            )
            rgb_geometry_usable = bool(
                cam0_path is not None
                and cam0_path.exists()
                and projected_bbox is not None
                and bbox_width >= args.minimum_rgb_size
                and bbox_height >= args.minimum_rgb_size
            )
            rgb_usable = bool(
                rgb_geometry_usable
                and (not mask_available or mask_pixel_count >= args.minimum_mask_pixels)
            )
            lidar_usable = bool(lidar_count >= args.minimum_lidar_points)
            if not rgb_usable and not lidar_usable:
                skipped_unobservable += 1
                continue

            axis_lengths = np.linalg.norm(box.transform_world[:3, :3], axis=0)
            transform_ego_object = velo_from_world @ box.transform_world
            box_local_min = box.vertices_local.min(axis=0)
            box_local_max = box.vertices_local.max(axis=0)
            instance_token = stable_token(
                "instance-v2", sequence, box.semantic_id, box.instance_id
            )
            annotation_token = stable_token("annotation-v2", str(sample.token), instance_token)
            unknown = box.raw_label.lower().startswith("unknown")
            annotation_rows.append(
                {
                    "token": annotation_token,
                    "sample_token": str(sample.token),
                    "scene_token": str(sample.scene_token),
                    "log_token": log_token,
                    "sequence": sequence,
                    "raw_frame_index": frame,
                    "timestamp_ns": int(sample.timestamp_ns),
                    "instance_token": instance_token,
                    "semantic_id": box.semantic_id,
                    "instance_id": box.instance_id,
                    "combined_instance_id": combined_instance_id,
                    "raw_label": box.raw_label,
                    "dynamic": box.dynamic,
                    "xml_timestamp": box.timestamp,
                    "xml_annotation_index": box.annotation_index,
                    "center_world": center_world.tolist(),
                    "center_ego": center_ego.tolist(),
                    "box_axis_lengths_m": axis_lengths.tolist(),
                    "box_volume_m3": float(np.prod(axis_lengths)),
                    "box_transform_world": box.transform_world.reshape(-1).tolist(),
                    "box_transform_ego": transform_ego_object.reshape(-1).tolist(),
                    "box_local_min": box_local_min.tolist(),
                    "box_local_max": box_local_max.tolist(),
                    "corners_world": corners_world.reshape(-1).tolist(),
                    "corners_ego": corners_ego.reshape(-1).tolist(),
                    "projected_bbox_xyxy": projected_bbox,
                    "projected_area_pixels": projected_area,
                    "camera_depth_m": camera_depth,
                    "mask_available": mask_available,
                    "mask_pixel_count": mask_pixel_count,
                    "lidar_point_count": lidar_count,
                    "rgb_usable": rgb_usable,
                    "lidar_usable": lidar_usable,
                    "modality_mask": [int(rgb_usable), int(lidar_usable)],
                    "supervised_eligible": not unknown,
                    "cam0_filename": cam0_relative,
                    "lidar_filename": lidar_relative,
                    "instance_mask_filename": mask_relative,
                    "source_xml": box.source_xml,
                    "split": str(sample.split),
                    "schema_version": SCHEMA_VERSION,
                }
            )
        if progress % 25 == 0 or progress == len(samples):
            print(
                f"Processed key sample {progress}/{len(samples)}; "
                f"kept {len(annotation_rows)} observations",
                flush=True,
            )

    annotations = pd.DataFrame(annotation_rows)
    if annotations.empty:
        raise RuntimeError("No usable object observations were produced")
    if not annotations["token"].is_unique:
        raise RuntimeError("Duplicate sample-annotation tokens detected")
    atomic_json_records(annotations, output_root / "sample_annotations.json")

    categories = (
        annotations.groupby(["semantic_id", "raw_label"], as_index=False)
        .agg(observation_count=("token", "size"), instance_count=("instance_token", "nunique"))
        .sort_values(["semantic_id", "raw_label"])
    )
    categories.insert(
        0,
        "token",
        [stable_token("category-v2", row.semantic_id, row.raw_label) for row in categories.itertuples()],
    )
    categories["schema_version"] = SCHEMA_VERSION
    atomic_json_records(categories, output_root / "categories.json")

    instances = (
        annotations.groupby(
            ["instance_token", "log_token", "sequence", "semantic_id", "instance_id", "raw_label"],
            as_index=False,
        )
        .agg(
            dynamic=("dynamic", "max"),
            first_frame=("raw_frame_index", "min"),
            last_frame=("raw_frame_index", "max"),
            observation_count=("token", "size"),
            rgb_observation_count=("rgb_usable", "sum"),
            lidar_observation_count=("lidar_usable", "sum"),
        )
        .sort_values(["sequence", "semantic_id", "instance_id"])
    )
    instances["schema_version"] = SCHEMA_VERSION
    atomic_json_records(instances, output_root / "instances.json")

    relation_path = output_root / "spatial_relations.json"
    predicate_counts, relation_count, no_relation_count, canonical_violations = (
        write_relation_json(annotations, relation_path, args)
    )

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "status": "provisional_candidate_protocol",
        "predicate_order": PREDICATES,
        "canonical_inverse_encoding": CANONICAL_INVERSE_ENCODING,
        "ordered_pair_policy": (
            "Every eligible ordered non-self pair is retained. Removed inverse names are "
            "represented by the canonical predicate on the reverse ordered edge."
        ),
        "coordinate_frame": "current Velodyne/ego frame: +x forward, +y left, +z up",
        "thresholds": {
            "minimum_rgb_size": args.minimum_rgb_size,
            "minimum_mask_pixels": args.minimum_mask_pixels,
            "minimum_lidar_points": args.minimum_lidar_points,
            "candidate_distance": args.candidate_distance,
            "relation_distance": args.relation_distance,
            "near_distance": args.near_distance,
            "lateral_min_margin": args.lateral_min_margin,
            "longitudinal_min_margin": args.longitudinal_min_margin,
            "overlap_bev_iou": args.overlap_bev_iou,
            "overlap_vertical": args.overlap_vertical,
            "occlusion_overlap": args.occlusion_overlap,
            "occlusion_depth_margin": args.occlusion_depth_margin,
        },
    }
    atomic_json(output_root / "predicate_config_provisional.json", configuration)

    report = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "counts": {
            "selected_key_samples": int(len(samples)),
            "xml_box_entries": len(boxes),
            "sample_annotations": int(len(annotations)),
            "distinct_instances": int(annotations["instance_token"].nunique()),
            "categories": int(len(categories)),
            "relation_candidates": relation_count,
            "no_relation_candidates": no_relation_count,
            "skipped_singular_boxes": skipped_singular,
            "skipped_unobservable_boxes": skipped_unobservable,
            "skipped_missing_pose_samples": skipped_missing_pose,
        },
        "predicate_positive_counts": predicate_counts,
        "validation": {
            "passed": len(canonical_violations) == 0,
            "unique_annotation_tokens": bool(annotations["token"].is_unique),
            "canonical_relation_violations": canonical_violations,
        },
        "warning": (
            "These are provisional geometry-derived labels for five canonical spatial "
            "predicates. Together with the four temporal predicates, the protocol has nine "
            "labels. Do not report final results until human review and threshold freezing."
        ),
    }
    atomic_json(result_file, report)
    atomic_json(
        output_root / "predicate_statistics.json",
        {
            "predicate_positive_counts": predicate_counts,
            "relation_candidates": relation_count,
            "no_relation_candidates": no_relation_count,
            "positive_rate_percent": {
                name: (100.0 * count / relation_count if relation_count else 0.0)
                for name, count in predicate_counts.items()
            },
        },
    )
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(predicate_counts, indent=2))
    print(json.dumps(report["validation"], indent=2))
    print(f"Saved predicate preparation to: {output_root}")
    if canonical_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
