from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .relation_data import ShardPair, decode_token, stack_tracklet


def _interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.float64)
    if valid.sum() == 1:
        return np.full_like(values, values[valid][0], dtype=np.float64)
    return np.interp(indices, indices[valid], values[valid]).astype(np.float64)


def _velocity(
    centers: np.ndarray, timestamps_ns: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    positions = np.flatnonzero(valid)
    if len(positions) < 2:
        return np.zeros(3, dtype=np.float64)
    first, last = int(positions[0]), int(positions[-1])
    duration = (int(timestamps_ns[last]) - int(timestamps_ns[first])) / 1e9
    if duration <= 0:
        return np.zeros(3, dtype=np.float64)
    return (centers[last] - centers[first]) / duration


def _point_segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    direction = second - first
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - first))
    fraction = float(np.clip(np.dot(point - first, direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + fraction * direction)))


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    first_direction = second - first
    second_direction = third - first
    return float(
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return bool(ab_c * ab_d <= 0.0 and cd_a * cd_b <= 0.0)


def _segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _polyline_distance(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) == 1 or len(second) == 1:
        return float(
            min(np.linalg.norm(left - right) for left in first for right in second)
        )
    return min(
        _segment_distance(first[i], first[i + 1], second[j], second[j + 1])
        for i in range(len(first) - 1)
        for j in range(len(second) - 1)
    )


def predicted_temporal_geometry(
    source_centers: np.ndarray,
    target_centers: np.ndarray,
    timestamps_ns: np.ndarray,
    valid_mask: np.ndarray,
    step_distance_epsilon: float = 0.05,
    minimum_motion_speed: float = 0.50,
) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if len(valid) != 5 or not valid[2]:
        raise ValueError("Predicted temporal geometry requires five steps and a valid middle.")
    relative = source_centers - target_centers
    raw_distances = np.linalg.norm(relative[:, :2], axis=1)
    distances = _interpolate(raw_distances, valid)
    distance_steps = np.diff(distances)
    distance_change = float(distances[-1] - distances[0])
    decreasing_fraction = float(np.mean(distance_steps <= -step_distance_epsilon))
    increasing_fraction = float(np.mean(distance_steps >= step_distance_epsilon))
    source_velocity = _velocity(source_centers, timestamps_ns, valid)
    target_velocity = _velocity(target_centers, timestamps_ns, valid)
    source_speed = float(np.linalg.norm(source_velocity[:2]))
    target_speed = float(np.linalg.norm(target_velocity[:2]))
    direction_cosine = 0.0
    if source_speed >= minimum_motion_speed and target_speed >= minimum_motion_speed:
        direction_cosine = float(
            np.clip(
                np.dot(source_velocity[:2], target_velocity[:2])
                / (source_speed * target_speed),
                -1.0,
                1.0,
            )
        )
    middle_relative = target_centers[2, :2] - source_centers[2, :2]
    middle_norm = float(np.linalg.norm(middle_relative))
    if middle_norm > 1e-6:
        direction = middle_relative / middle_norm
        source_radial = float(np.dot(source_velocity[:2], direction))
        target_radial = float(np.dot(target_velocity[:2], -direction))
    else:
        source_radial = target_radial = 0.0
    observed_path = 0.0
    if source_speed >= minimum_motion_speed and target_speed >= minimum_motion_speed:
        observed_path = _polyline_distance(
            source_centers[valid, :2], target_centers[valid, :2]
        )
    visible_fraction = float(valid.mean())
    return np.asarray(
        [
            *distances.tolist(),
            distance_change,
            decreasing_fraction,
            increasing_fraction,
            source_speed,
            target_speed,
            *source_velocity.tolist(),
            *target_velocity.tolist(),
            source_radial,
            target_radial,
            direction_cosine,
            float(distances[2]),
            visible_fraction,
            visible_fraction,
            visible_fraction,
            observed_path,
        ],
        dtype=np.float32,
    )


class PredictedShard:
    def __init__(self, feature_path: Path, relation_path: Path, track_path: Path):
        self.pair = ShardPair(feature_path, relation_path)
        self.track_path = track_path.resolve()
        self.tracks: h5py.File | None = None
        if not self.track_path.is_file():
            raise FileNotFoundError(f"Predicted-tracklet shard not found: {self.track_path}")
        with h5py.File(self.track_path, "r") as tracks:
            if tracks.attrs.get("schema_version", "") != (
                "IBP-K360-predicted-tracklets-v1.1.0"
            ):
                raise ValueError("Predicted Stage 3 requires predicted-tracklet schema v1.1.0.")
        with h5py.File(self.pair.relation_path, "r") as relations:
            if relations.attrs.get("schema_version", "") != "IBP-K360-relation-index-v1.2.0":
                raise ValueError("Predicted Stage 3 requires relation-index schema v1.2.0.")
            self.length = len(relations["temporal"]["targets"])
        self._members: dict[tuple[int, int], int] | None = None
        self._frame_timestamps: dict[int, int] | None = None

    def open(self) -> tuple[h5py.File, h5py.File, h5py.File]:
        features, relations = self.pair.open()
        if self.tracks is None:
            self.tracks = h5py.File(self.track_path, "r", swmr=True)
        return features, relations, self.tracks

    def members(self) -> dict[tuple[int, int], int]:
        if self._members is None:
            _, _, tracks = self.open()
            track_ids = np.asarray(tracks["row_track_ids"], dtype=np.int64)
            raw_frames = np.asarray(tracks["row_raw_frame_indices"], dtype=np.int64)
            self._members = {}
            for row, (track_id, raw_frame) in enumerate(zip(track_ids, raw_frames)):
                if track_id < 0:
                    continue
                key = (int(track_id), int(raw_frame))
                if key in self._members:
                    raise ValueError(f"Predicted track has duplicate frame membership: {key}")
                self._members[key] = row
        return self._members

    def frame_timestamps(self) -> dict[int, int]:
        if self._frame_timestamps is None:
            _, _, tracks = self.open()
            self._frame_timestamps = {}
            for raw_frame, timestamp in zip(
                tracks["timeline_raw_frame_indices"], tracks["timeline_timestamps_ns"]
            ):
                self._frame_timestamps[int(raw_frame)] = int(timestamp)
        return self._frame_timestamps

    def predicted_rows(self, relation_row: int) -> tuple[np.ndarray, np.ndarray, bool]:
        _, relations, tracks = self.open()
        group = relations["temporal"]
        frames = np.asarray(group["raw_frame_indices"][relation_row], dtype=np.int64)
        source_middle = int(group["source_rows"][relation_row, 2])
        target_middle = int(group["target_rows"][relation_row, 2])
        track_ids = tracks["row_track_ids"]
        if source_middle < 0 or target_middle < 0:
            return np.full(5, -1), np.full(5, -1), False
        source_track = int(track_ids[source_middle])
        target_track = int(track_ids[target_middle])
        member_lookup = self.members()
        source_rows = np.asarray(
            [member_lookup.get((source_track, int(frame)), -1) for frame in frames],
            dtype=np.int64,
        )
        target_rows = np.asarray(
            [member_lookup.get((target_track, int(frame)), -1) for frame in frames],
            dtype=np.int64,
        )
        joint = (source_rows >= 0) & (target_rows >= 0)
        available = bool(
            source_track >= 0
            and target_track >= 0
            and source_track != target_track
            and joint[2]
            and joint.sum() >= 3
        )
        return source_rows, target_rows, available

    def item(self, relation_row: int) -> dict[str, Any]:
        features, relations, tracks = self.open()
        group = relations["temporal"]
        source_rows, target_rows, available = self.predicted_rows(relation_row)
        joint_mask = (source_rows >= 0) & (target_rows >= 0)
        if not joint_mask[2]:
            # The fixed relation candidate always has a valid middle observation.
            source_rows = np.asarray(group["source_rows"][relation_row], dtype=np.int64)
            target_rows = np.asarray(group["target_rows"][relation_row], dtype=np.int64)
            joint_mask = np.zeros(5, dtype=np.bool_)
            joint_mask[2] = source_rows[2] >= 0 and target_rows[2] >= 0
        source_centers = np.zeros((5, 3), dtype=np.float64)
        target_centers = np.zeros((5, 3), dtype=np.float64)
        for position in np.flatnonzero(joint_mask):
            source_centers[position] = features["center_world"][int(source_rows[position])]
            target_centers[position] = features["center_world"][int(target_rows[position])]
        if joint_mask.sum() == 1:
            source_centers[:] = source_centers[joint_mask][0]
            target_centers[:] = target_centers[joint_mask][0]
        else:
            for dimension in range(3):
                source_centers[:, dimension] = _interpolate(
                    source_centers[:, dimension], joint_mask
                )
                target_centers[:, dimension] = _interpolate(
                    target_centers[:, dimension], joint_mask
                )
        frames = np.asarray(group["raw_frame_indices"][relation_row], dtype=np.int64)
        frame_timestamps = self.frame_timestamps()
        timestamps = np.asarray([frame_timestamps[int(frame)] for frame in frames], dtype=np.int64)
        safe_source = np.where(joint_mask, source_rows, source_rows[2])
        safe_target = np.where(joint_mask, target_rows, target_rows[2])
        return {
            "source": stack_tracklet(self.pair, safe_source, joint_mask),
            "target": stack_tracklet(self.pair, safe_target, joint_mask),
            "geometry": torch.from_numpy(
                predicted_temporal_geometry(
                    source_centers, target_centers, timestamps, joint_mask
                )
            ).float(),
            "labels": torch.from_numpy(group["targets"][relation_row]).float(),
            "available": available,
            "relation_id": decode_token(group["relation_ids"][relation_row]),
            "group_id": decode_token(group["group_ids"][relation_row]),
            "sequence": decode_token(relations.attrs.get("sequence", "")),
            "predicted_source_rows": torch.from_numpy(source_rows),
            "predicted_target_rows": torch.from_numpy(target_rows),
            "predicted_joint_mask": torch.from_numpy(joint_mask),
        }

    def close(self) -> None:
        self.pair.close()
        if self.tracks is not None:
            self.tracks.close()
            self.tracks = None
        self._members = None
        self._frame_timestamps = None


class PredictedTemporalRelationDataset(Dataset):
    def __init__(
        self,
        feature_paths: list[Path],
        relation_paths: list[Path],
        track_paths: list[Path],
    ):
        if not feature_paths or not (
            len(feature_paths) == len(relation_paths) == len(track_paths)
        ):
            raise ValueError("Feature, relation, and predicted-tracklet shards must align.")
        self.shards = [
            PredictedShard(feature, relation, track)
            for feature, relation, track in zip(feature_paths, relation_paths, track_paths)
        ]
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += shard.length
        self.total = total
        self.eligible_indices: list[int] = []
        self.eligible_by_shard: list[list[int]] = []
        self.availability_by_shard: list[dict[str, int]] = []
        for offset, shard in zip(self.offsets, self.shards):
            available = 0
            local_eligible: list[int] = []
            for row in range(shard.length):
                _, _, usable = shard.predicted_rows(row)
                if usable:
                    self.eligible_indices.append(offset + row)
                    local_eligible.append(row)
                    available += 1
            self.eligible_by_shard.append(local_eligible)
            self.availability_by_shard.append(
                {"total": shard.length, "available": available, "unavailable": shard.length - available}
            )

    def __len__(self) -> int:
        return self.total

    def locate(self, index: int) -> tuple[PredictedShard, int]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        for shard_index in range(len(self.shards) - 1, -1, -1):
            if index >= self.offsets[shard_index]:
                return self.shards[shard_index], index - self.offsets[shard_index]
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, row = self.locate(index)
        return shard.item(row)

    def close(self) -> None:
        for shard in self.shards:
            shard.close()

    def __del__(self):
        self.close()
