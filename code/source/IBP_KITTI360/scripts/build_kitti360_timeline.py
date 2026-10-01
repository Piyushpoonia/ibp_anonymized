#!/usr/bin/env python3
"""Build a nuScenes-style KITTI-360 timeline without object preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


SCHEMA_VERSION = "IBP-K360-timeline-v3.0.0"
CHANNELS = ("CAM0", "CAM1", "LIDAR_TOP", "SEMANTIC_CAM0", "INSTANCE_CAM0")


def stable_token(kind: str, *parts: object) -> str:
    value = "|".join([SCHEMA_VERSION, kind, *(str(part) for part in parts)])
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:32]


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def matrix4(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 12:
        result = np.eye(4, dtype=np.float64)
        result[:3, :] = values.reshape(3, 4)
        return result
    if values.size == 16:
        return values.reshape(4, 4)
    raise ValueError(f"Expected 12 or 16 matrix values, received {values.size}")


def flat_matrix(matrix: np.ndarray) -> list[float]:
    return matrix.astype(np.float64).reshape(-1).tolist()


def read_keyed_calibration(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = [float(item) for item in raw.split()]
        except ValueError:
            continue
    return values


def read_single_matrix(path: Path) -> np.ndarray:
    raw = [float(item) for item in path.read_text(encoding="utf-8").split()]
    return matrix4(raw)


def read_pose_map(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            frame = int(fields[0])
            poses[frame] = matrix4([float(value) for value in fields[1:]])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid pose at {path}:{line_number}") from exc
    return poses


def read_timestamps(path: Path) -> tuple[list[str], np.ndarray]:
    text = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    nanoseconds = np.empty(len(text), dtype=np.int64)
    for index, value in enumerate(text):
        try:
            nanoseconds[index] = np.datetime64(value.replace(" ", "T"), "ns").astype(np.int64)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp at {path}:{index + 1}: {value}") from exc
    return text, nanoseconds


def indexed_files(folder: Path, suffix: str) -> dict[int, Path]:
    if not folder.exists():
        return {}
    result: dict[int, Path] = {}
    for path in folder.glob(f"*{suffix}"):
        try:
            result[int(path.stem)] = path
        except ValueError:
            continue
    return result


def contiguous_runs(frame_ids: Sequence[int]) -> list[list[int]]:
    if not frame_ids:
        return []
    runs: list[list[int]] = []
    current = [int(frame_ids[0])]
    for frame in frame_ids[1:]:
        frame = int(frame)
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            runs.append(current)
            current = [frame]
    runs.append(current)
    return runs


def load_split_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    if all(key in {"train", "validation", "test"} for key in payload):
        for split, sequences in payload.items():
            for sequence in sequences:
                if sequence in result:
                    raise ValueError(f"Sequence appears in multiple splits: {sequence}")
                result[str(sequence)] = split
    else:
        result = {str(sequence): str(split) for sequence, split in payload.items()}
    invalid = sorted(set(result.values()) - {"train", "validation", "test"})
    if invalid:
        raise ValueError(f"Invalid split names: {invalid}")
    return result


@dataclass
class PoseResult:
    matrix: np.ndarray | None
    source: str
    interpolation_span_frames: int | None


class PoseProvider:
    def __init__(self, poses: dict[int, np.ndarray]):
        if not poses:
            self.frames = np.empty(0, dtype=np.int64)
            self.poses = poses
            return
        self.frames = np.asarray(sorted(poses), dtype=np.int64)
        self.poses = poses

    def get(self, frame: int) -> PoseResult:
        exact = self.poses.get(frame)
        if exact is not None:
            return PoseResult(exact, "exact", 0)
        if self.frames.size < 2:
            return PoseResult(None, "missing", None)
        position = int(np.searchsorted(self.frames, frame))
        if position == 0 or position == self.frames.size:
            return PoseResult(None, "missing", None)
        left_frame = int(self.frames[position - 1])
        right_frame = int(self.frames[position])
        left = self.poses[left_frame]
        right = self.poses[right_frame]
        alpha = (frame - left_frame) / float(right_frame - left_frame)
        translation = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
        rotations = Rotation.from_matrix(np.stack([left[:3, :3], right[:3, :3]]))
        rotation = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation
        result[:3, 3] = translation
        return PoseResult(result, "interpolated", right_frame - left_frame)


def calibrated_sensor_rows(root: Path) -> tuple[list[dict], dict[str, str], dict[str, np.ndarray]]:
    calibration = root / "calibration"
    perspective = read_keyed_calibration(calibration / "perspective.txt")
    cam_to_pose = read_keyed_calibration(calibration / "calib_cam_to_pose.txt")
    t_velo_cam0_raw = read_single_matrix(calibration / "calib_cam_to_velo.txt")

    t_pose_cam0 = matrix4(cam_to_pose["image_00"])
    t_pose_cam1 = matrix4(cam_to_pose["image_01"])
    t_cam0_cam1 = np.linalg.inv(t_pose_cam0) @ t_pose_cam1
    t_velo_cam1_raw = t_velo_cam0_raw @ t_cam0_cam1

    r_rect_00 = np.eye(4, dtype=np.float64)
    r_rect_00[:3, :3] = np.asarray(perspective["R_rect_00"]).reshape(3, 3)
    r_rect_01 = np.eye(4, dtype=np.float64)
    r_rect_01[:3, :3] = np.asarray(perspective["R_rect_01"]).reshape(3, 3)
    t_velo_cam0_rect = t_velo_cam0_raw @ np.linalg.inv(r_rect_00)
    t_velo_cam1_rect = t_velo_cam1_raw @ np.linalg.inv(r_rect_01)

    matrices = {
        "T_velo_cam0_raw": t_velo_cam0_raw,
        "T_velo_cam0_rect": t_velo_cam0_rect,
        "T_velo_cam1_rect": t_velo_cam1_rect,
    }
    tokens = {channel: stable_token("calibrated_sensor", channel) for channel in CHANNELS}
    rows: list[dict] = []
    camera_specs = {
        "CAM0": (t_velo_cam0_rect, "P_rect_00", "K_00", "D_00", "S_rect_00"),
        "CAM1": (t_velo_cam1_rect, "P_rect_01", "K_01", "D_01", "S_rect_01"),
    }
    for channel, (transform, p_key, k_key, d_key, size_key) in camera_specs.items():
        size = perspective[size_key]
        rows.append(
            {
                "token": tokens[channel],
                "channel": channel,
                "modality": "camera",
                "T_ego_sensor": flat_matrix(transform),
                "T_sensor_ego": flat_matrix(np.linalg.inv(transform)),
                "camera_intrinsic": perspective[k_key],
                "projection_matrix": perspective[p_key],
                "distortion": perspective[d_key],
                "width": int(size[0]),
                "height": int(size[1]),
                "source_calibration": "KITTI-360 perspective rectified",
                "schema_version": SCHEMA_VERSION,
            }
        )
    rows.append(
        {
            "token": tokens["LIDAR_TOP"],
            "channel": "LIDAR_TOP",
            "modality": "lidar",
            "T_ego_sensor": flat_matrix(np.eye(4)),
            "T_sensor_ego": flat_matrix(np.eye(4)),
            "camera_intrinsic": None,
            "projection_matrix": None,
            "distortion": None,
            "width": None,
            "height": None,
            "source_calibration": "ego frame",
            "schema_version": SCHEMA_VERSION,
        }
    )
    for channel in ("SEMANTIC_CAM0", "INSTANCE_CAM0"):
        base = dict(next(row for row in rows if row["channel"] == "CAM0"))
        base.update({"token": tokens[channel], "channel": channel, "modality": "label"})
        rows.append(base)
    return rows, tokens, matrices


def atomic_parquet(rows: list[dict], path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists; use --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: object, path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists; use --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def source_entry(path: Path, root: Path) -> str:
    return f"{relative_posix(path, root)}|{path.stat().st_size}"


def select_sequences(root: Path, requested: Sequence[str] | None) -> list[str]:
    rgb = {path.name for path in (root / "data_2d_raw").glob("*_sync") if path.is_dir()}
    lidar = {path.name for path in (root / "data_3d_raw").glob("*_sync") if path.is_dir()}
    poses = {path.name for path in (root / "data_poses").glob("*_sync") if path.is_dir()}
    available = sorted(rgb & lidar & poses)
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(f"Requested sequences are incomplete or missing: {missing}")
        return sorted(set(requested))
    return available


def validate_tables(
    log_rows: list[dict],
    scene_rows: list[dict],
    sample_rows: list[dict],
    raw_frame_rows: list[dict],
    sample_data_rows: list[dict],
    pose_rows: list[dict],
    sensor_rows: list[dict],
    scene_frames: int,
    sample_stride: int,
) -> dict:
    errors: list[str] = []

    def unique(rows: list[dict], field: str, label: str) -> set[str]:
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            errors.append(f"Duplicate {label} {field}")
        return set(values)

    log_tokens = unique(log_rows, "token", "log")
    scene_tokens = unique(scene_rows, "token", "scene")
    sample_tokens = unique(sample_rows, "token", "sample")
    raw_frame_tokens = unique(raw_frame_rows, "token", "raw frame")
    pose_tokens = unique(pose_rows, "token", "ego pose")
    sensor_tokens = unique(sensor_rows, "token", "calibrated sensor")
    unique(sample_data_rows, "token", "sample data")

    if any(row["log_token"] not in log_tokens for row in scene_rows):
        errors.append("Scene references an unknown log")
    if any(row["scene_token"] not in scene_tokens for row in sample_rows):
        errors.append("Sample references an unknown scene")
    if any(row["scene_token"] not in scene_tokens for row in raw_frame_rows):
        errors.append("Raw frame references an unknown scene")
    if any(row["sample_token"] not in sample_tokens for row in sample_data_rows):
        errors.append("Sample data references an unknown sample")
    if any(row["raw_frame_token"] not in raw_frame_tokens for row in sample_data_rows):
        errors.append("Sample data references an unknown raw frame")
    if any(row["calibrated_sensor_token"] not in sensor_tokens for row in sample_data_rows):
        errors.append("Sample data references an unknown calibrated sensor")
    if any(row["ego_pose_token"] and row["ego_pose_token"] not in pose_tokens for row in sample_data_rows):
        errors.append("Sample data references an unknown ego pose")

    expected_complete_samples = int(math.ceil(scene_frames / sample_stride))
    bad_complete = [
        row["token"]
        for row in scene_rows
        if row["complete"] and row["nbr_samples"] != expected_complete_samples
    ]
    if bad_complete:
        errors.append(f"Complete scenes with wrong sample count: {len(bad_complete)}")

    scenes_by_log: dict[str, list[dict]] = {}
    for row in scene_rows:
        scenes_by_log.setdefault(row["log_token"], []).append(row)
    for rows in scenes_by_log.values():
        rows.sort(key=lambda item: item["raw_start_frame"])
        for previous, current in zip(rows, rows[1:]):
            if current["raw_start_frame"] <= previous["raw_end_frame"]:
                errors.append("Scene clips overlap")
                break

    samples_by_scene: dict[str, list[dict]] = {}
    for row in sample_rows:
        samples_by_scene.setdefault(row["scene_token"], []).append(row)
    for rows in samples_by_scene.values():
        rows.sort(key=lambda item: item["sample_index"])
        frame_ids = [row["raw_frame_index"] for row in rows]
        if any(right - left != sample_stride for left, right in zip(frame_ids, frame_ids[1:])):
            errors.append("Sample stride differs from the configured stride")
            break

    return {
        "passed": not errors,
        "errors": errors,
        "checks": {
            "unique_tokens": not any("Duplicate" in error for error in errors),
            "foreign_keys_resolve": not any("unknown" in error for error in errors),
            "complete_scene_sample_count": not any("wrong sample count" in error for error in errors),
            "non_overlapping_scenes": "Scene clips overlap" not in errors,
            "fixed_sample_stride": not any("Sample stride" in error for error in errors),
        },
    }


def build(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    output = args.output_root.resolve()
    required = [
        root / "calibration/perspective.txt",
        root / "calibration/calib_cam_to_pose.txt",
        root / "calibration/calib_cam_to_velo.txt",
        root / "data_2d_raw",
        root / "data_3d_raw",
        root / "data_poses",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required KITTI-360 inputs:\n" + "\n".join(missing))

    sequences = select_sequences(root, args.sequences)
    if not sequences:
        raise RuntimeError("No complete KITTI-360 sequences found")
    split_map = load_split_map(args.split_file)
    sensor_rows, sensor_tokens, calibration_matrices = calibrated_sensor_rows(root)
    t_cam0_velo = np.linalg.inv(calibration_matrices["T_velo_cam0_raw"])

    log_rows: list[dict] = []
    scene_rows: list[dict] = []
    sample_rows: list[dict] = []
    raw_frame_rows: list[dict] = []
    sample_data_rows: list[dict] = []
    pose_rows: list[dict] = []
    rejected_segments: list[dict] = []
    sequence_reports: dict[str, dict] = {}

    for sequence in sequences:
        print(f"Scanning {sequence}...", flush=True)
        cam0_dir = root / f"data_2d_raw/{sequence}/image_00/data_rect"
        cam1_dir = root / f"data_2d_raw/{sequence}/image_01/data_rect"
        lidar_dir = root / f"data_3d_raw/{sequence}/velodyne_points/data"
        semantic_dir = root / f"data_2d_semantics/train/{sequence}/image_00/semantic"
        instance_dir = root / f"data_2d_semantics/train/{sequence}/image_00/instance"

        cam0 = indexed_files(cam0_dir, ".png")
        cam1 = indexed_files(cam1_dir, ".png")
        lidar = indexed_files(lidar_dir, ".bin")
        semantic = indexed_files(semantic_dir, ".png")
        instance = indexed_files(instance_dir, ".png")
        synchronized = sorted(set(cam0) & set(lidar))
        if not synchronized:
            rejected_segments.append({"sequence": sequence, "reason": "no_synchronized_frames"})
            continue

        cam0_timestamp_text, cam0_timestamps = read_timestamps(
            root / f"data_2d_raw/{sequence}/image_00/timestamps.txt"
        )
        cam1_timestamp_path = root / f"data_2d_raw/{sequence}/image_01/timestamps.txt"
        if cam1_timestamp_path.exists():
            cam1_timestamp_text, cam1_timestamps = read_timestamps(cam1_timestamp_path)
        else:
            cam1_timestamp_text, cam1_timestamps = cam0_timestamp_text, cam0_timestamps
        lidar_timestamp_text, lidar_timestamps = read_timestamps(
            root / f"data_3d_raw/{sequence}/velodyne_points/timestamps.txt"
        )
        max_frame = max(synchronized)
        if max_frame >= len(cam0_timestamps) or max_frame >= len(lidar_timestamps):
            raise IndexError(
                f"Timestamp files for {sequence} do not cover frame {max_frame}"
            )

        pose_path = root / f"data_poses/{sequence}/cam0_to_world.txt"
        if not pose_path.exists():
            pose_path = root / f"data_poses/{sequence}/poses.txt"
        pose_provider = PoseProvider(read_pose_map(pose_path))

        timestamp_values = lidar_timestamps[np.asarray(synchronized, dtype=np.int64)]
        positive_deltas = np.diff(timestamp_values)
        positive_deltas = positive_deltas[positive_deltas > 0]
        median_period_ns = int(np.median(positive_deltas)) if positive_deltas.size else 100_000_000
        nominal_fps = float(1_000_000_000.0 / median_period_ns)
        log_token = stable_token("log", sequence)
        split = split_map.get(sequence, "unassigned")
        manifest_hasher = hashlib.sha256()
        accepted_scene_tokens: list[str] = []
        scene_counter = 0
        sequence_raw_count = 0
        sequence_sample_count = 0
        exact_pose_count = 0
        interpolated_pose_count = 0
        missing_pose_count = 0

        runs = contiguous_runs(synchronized)
        for run_index, run in enumerate(runs):
            cursor = 0
            while cursor < len(run):
                chunk = run[cursor : cursor + args.scene_frames]
                complete = len(chunk) == args.scene_frames
                if not complete and len(chunk) < args.min_tail_frames:
                    rejected_segments.append(
                        {
                            "sequence": sequence,
                            "run_index": run_index,
                            "raw_start_frame": chunk[0],
                            "raw_end_frame": chunk[-1],
                            "raw_frame_count": len(chunk),
                            "reason": "short_tail",
                        }
                    )
                    break
                if args.max_scenes_per_log is not None and scene_counter >= args.max_scenes_per_log:
                    rejected_segments.append(
                        {
                            "sequence": sequence,
                            "run_index": run_index,
                            "raw_start_frame": chunk[0],
                            "raw_end_frame": run[-1],
                            "raw_frame_count": len(run) - cursor,
                            "reason": "development_scene_limit",
                        }
                    )
                    cursor = len(run)
                    break

                scene_token = stable_token("scene", sequence, scene_counter, chunk[0], chunk[-1])
                accepted_scene_tokens.append(scene_token)
                key_frames = chunk[:: args.sample_stride]
                key_tokens = [stable_token("sample", sequence, frame) for frame in key_frames]
                sample_for_frame = {
                    frame: key_tokens[min(index // args.sample_stride, len(key_tokens) - 1)]
                    for index, frame in enumerate(chunk)
                }
                scene_start_ns = int(lidar_timestamps[chunk[0]])
                scene_end_ns = int(lidar_timestamps[chunk[-1]])

                for sample_index, (frame, sample_token) in enumerate(zip(key_frames, key_tokens)):
                    sample_rows.append(
                        {
                            "token": sample_token,
                            "scene_token": scene_token,
                            "log_token": log_token,
                            "timestamp_ns": int(lidar_timestamps[frame]),
                            "timestamp_text": lidar_timestamp_text[frame],
                            "raw_frame_index": frame,
                            "sample_index": sample_index,
                            "prev": key_tokens[sample_index - 1] if sample_index else "",
                            "next": key_tokens[sample_index + 1] if sample_index + 1 < len(key_tokens) else "",
                            "split": split,
                            "has_semantic": frame in semantic,
                            "has_instance": frame in instance,
                            "schema_version": SCHEMA_VERSION,
                        }
                    )

                channel_record_indices: dict[str, list[int]] = {channel: [] for channel in CHANNELS}
                for raw_index, frame in enumerate(chunk):
                    raw_frame_token = stable_token("raw_frame", sequence, frame)
                    sample_token = sample_for_frame[frame]
                    is_key_frame = frame in set(key_frames)
                    pose_result = pose_provider.get(frame)
                    ego_pose_token = ""
                    if pose_result.matrix is None:
                        missing_pose_count += 1
                    else:
                        ego_pose_token = stable_token("ego_pose", sequence, frame)
                        t_world_camera0 = pose_result.matrix
                        t_world_velo = t_world_camera0 @ t_cam0_velo
                        rotation_xyzw = Rotation.from_matrix(t_world_velo[:3, :3]).as_quat().tolist()
                        pose_rows.append(
                            {
                                "token": ego_pose_token,
                                "raw_frame_token": raw_frame_token,
                                "sample_token": sample_token,
                                "scene_token": scene_token,
                                "log_token": log_token,
                                "sequence": sequence,
                                "raw_frame_index": frame,
                                "timestamp_ns": int(lidar_timestamps[frame]),
                                "translation_world": t_world_velo[:3, 3].tolist(),
                                "rotation_world_xyzw": rotation_xyzw,
                                "T_world_velo": flat_matrix(t_world_velo),
                                "T_world_camera0": flat_matrix(t_world_camera0),
                                "pose_source": pose_result.source,
                                "interpolation_span_frames": pose_result.interpolation_span_frames,
                                "pose_source_path": relative_posix(pose_path, root),
                                "schema_version": SCHEMA_VERSION,
                            }
                        )
                        if pose_result.source == "exact":
                            exact_pose_count += 1
                        else:
                            interpolated_pose_count += 1

                    raw_frame_rows.append(
                        {
                            "token": raw_frame_token,
                            "log_token": log_token,
                            "scene_token": scene_token,
                            "sample_token": sample_token,
                            "sequence": sequence,
                            "raw_frame_index": frame,
                            "raw_index_in_scene": raw_index,
                            "timestamp_ns": int(lidar_timestamps[frame]),
                            "timestamp_text": lidar_timestamp_text[frame],
                            "is_key_frame": is_key_frame,
                            "split": split,
                            "has_pose": bool(ego_pose_token),
                            "has_cam0": frame in cam0,
                            "has_cam1": frame in cam1,
                            "has_lidar": frame in lidar,
                            "has_semantic": frame in semantic,
                            "has_instance": frame in instance,
                            "schema_version": SCHEMA_VERSION,
                        }
                    )

                    available_channels = [
                        ("CAM0", cam0.get(frame), cam0_timestamp_text, cam0_timestamps),
                        ("CAM1", cam1.get(frame), cam1_timestamp_text, cam1_timestamps),
                        ("LIDAR_TOP", lidar.get(frame), lidar_timestamp_text, lidar_timestamps),
                        ("SEMANTIC_CAM0", semantic.get(frame), cam0_timestamp_text, cam0_timestamps),
                        ("INSTANCE_CAM0", instance.get(frame), cam0_timestamp_text, cam0_timestamps),
                    ]
                    for channel, path, timestamp_text, timestamps in available_channels:
                        if path is None:
                            continue
                        token = stable_token("sample_data", sequence, frame, channel)
                        manifest_hasher.update(source_entry(path, root).encode("utf-8"))
                        row = {
                            "token": token,
                            "sample_token": sample_token,
                            "raw_frame_token": raw_frame_token,
                            "scene_token": scene_token,
                            "log_token": log_token,
                            "ego_pose_token": ego_pose_token,
                            "calibrated_sensor_token": sensor_tokens[channel],
                            "channel": channel,
                            "modality": "camera" if channel.startswith("CAM") else "lidar" if channel == "LIDAR_TOP" else "label",
                            "timestamp_ns": int(timestamps[frame]),
                            "timestamp_text": timestamp_text[frame],
                            "filename": relative_posix(path, root),
                            "file_size_bytes": path.stat().st_size,
                            "is_key_frame": is_key_frame,
                            "prev": "",
                            "next": "",
                            "schema_version": SCHEMA_VERSION,
                        }
                        channel_record_indices[channel].append(len(sample_data_rows))
                        sample_data_rows.append(row)

                for indices in channel_record_indices.values():
                    for position, row_index in enumerate(indices):
                        sample_data_rows[row_index]["prev"] = (
                            sample_data_rows[indices[position - 1]]["token"] if position else ""
                        )
                        sample_data_rows[row_index]["next"] = (
                            sample_data_rows[indices[position + 1]]["token"]
                            if position + 1 < len(indices)
                            else ""
                        )

                scene_rows.append(
                    {
                        "token": scene_token,
                        "log_token": log_token,
                        "name": f"{sequence}_scene_{scene_counter:04d}",
                        "clip_index": scene_counter,
                        "run_index": run_index,
                        "raw_start_frame": chunk[0],
                        "raw_end_frame": chunk[-1],
                        "raw_frame_count": len(chunk),
                        "start_timestamp_ns": scene_start_ns,
                        "end_timestamp_ns": scene_end_ns,
                        "duration_sec": (scene_end_ns - scene_start_ns) / 1_000_000_000.0,
                        "first_sample_token": key_tokens[0],
                        "last_sample_token": key_tokens[-1],
                        "nbr_samples": len(key_tokens),
                        "complete": complete,
                        "split": split,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                sequence_raw_count += len(chunk)
                sequence_sample_count += len(key_frames)
                scene_counter += 1
                cursor += len(chunk)

        log_rows.append(
            {
                "token": log_token,
                "name": sequence,
                "description": "KITTI-360 synchronized driving log",
                "split": split,
                "first_raw_frame": synchronized[0],
                "last_raw_frame": synchronized[-1],
                "synchronized_frame_count": len(synchronized),
                "accepted_raw_frame_count": sequence_raw_count,
                "nominal_fps": nominal_fps,
                "scene_count": scene_counter,
                "first_scene_token": accepted_scene_tokens[0] if accepted_scene_tokens else "",
                "last_scene_token": accepted_scene_tokens[-1] if accepted_scene_tokens else "",
                "source_manifest_sha256": manifest_hasher.hexdigest(),
                "schema_version": SCHEMA_VERSION,
            }
        )
        sequence_reports[sequence] = {
            "synchronized_frames": len(synchronized),
            "contiguous_runs": len(runs),
            "accepted_raw_frames": sequence_raw_count,
            "scenes": scene_counter,
            "key_samples": sequence_sample_count,
            "exact_poses": exact_pose_count,
            "interpolated_poses": interpolated_pose_count,
            "missing_poses": missing_pose_count,
            "semantic_files": len(semantic),
            "instance_files": len(instance),
            "nominal_fps": nominal_fps,
        }
        print(json.dumps({sequence: sequence_reports[sequence]}, indent=2), flush=True)

    validation = validate_tables(
        log_rows,
        scene_rows,
        sample_rows,
        raw_frame_rows,
        sample_data_rows,
        pose_rows,
        sensor_rows,
        args.scene_frames,
        args.sample_stride,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "output_root": str(output),
        "parameters": {
            "scene_frames": args.scene_frames,
            "sample_stride": args.sample_stride,
            "min_tail_frames": args.min_tail_frames,
            "max_scenes_per_log": args.max_scenes_per_log,
        },
        "counts": {
            "logs": len(log_rows),
            "scenes": len(scene_rows),
            "samples": len(sample_rows),
            "raw_frames": len(raw_frame_rows),
            "sample_data": len(sample_data_rows),
            "ego_poses": len(pose_rows),
            "rejected_segments": len(rejected_segments),
        },
        "sequences": sequence_reports,
        "rejected_segments": rejected_segments,
        "validation": validation,
    }

    table_root = output / "tables"
    atomic_parquet(log_rows, table_root / "log.parquet", args.force)
    atomic_parquet(scene_rows, table_root / "scene.parquet", args.force)
    atomic_parquet(sample_rows, table_root / "sample.parquet", args.force)
    atomic_parquet(raw_frame_rows, table_root / "raw_frame.parquet", args.force)
    atomic_parquet(sample_data_rows, table_root / "sample_data.parquet", args.force)
    atomic_parquet(pose_rows, table_root / "ego_pose.parquet", args.force)
    atomic_parquet(sensor_rows, table_root / "calibrated_sensor.parquet", args.force)
    atomic_json(report, output / "scene_build_report.json", args.force)
    atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_root": str(root),
            "coordinate_frame": "ego=Velodyne (+x forward, +y left, +z up)",
            "quaternion_order": "xyzw",
            "matrix_layout": "row-major",
            "parameters": report["parameters"],
        },
        output / "preprocessing_config.json",
        args.force,
    )

    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(validation, indent=2))
    print(f"Saved timeline to: {output}")
    if not validation["passed"]:
        raise RuntimeError("Timeline validation failed; inspect scene_build_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-frames", type=int, default=200)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--min-tail-frames", type=int, default=100)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--max-scenes-per-log", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.scene_frames <= 0 or args.sample_stride <= 0 or args.min_tail_frames <= 0:
        parser.error("Scene, stride and tail parameters must be positive")
    if args.min_tail_frames > args.scene_frames:
        parser.error("--min-tail-frames cannot exceed --scene-frames")
    return args


if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
