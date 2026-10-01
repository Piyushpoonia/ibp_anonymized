#!/usr/bin/env python3
"""Build five-observation KITTI-360 tracklets and four temporal predicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString


SCHEMA_VERSION = "IBP-K360-temporal-predicates-v2.0.1-candidate"
TEMPORAL_PREDICATES = [
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
]
TEMPORAL_RELATION_COLUMNS = [
    "token",
    "scene_token",
    "sequence",
    "start_sample_index",
    "middle_sample_index",
    "middle_raw_frame_index",
    "raw_frame_indices",
    "rgb_left_filenames",
    "subject_tracklet_token",
    "object_tracklet_token",
    "subject_annotation_tokens",
    "object_annotation_tokens",
    "subject_instance_token",
    "object_instance_token",
    "subject_label",
    "object_label",
    "subject_projected_bboxes_xyxy",
    "object_projected_bboxes_xyxy",
    "subject_speed_mps",
    "object_speed_mps",
    "relative_distances_m",
    "distance_change_m",
    "decreasing_step_fraction",
    "increasing_step_fraction",
    "motion_direction_cosine",
    "observed_path_distance_m",
    "middle_distance_m",
    "predicate_vector",
    "positive_predicates",
    "no_relation",
    "supervised_eligible",
    "schema_version",
    *TEMPORAL_PREDICATES,
]


def stable_token(*parts: object) -> str:
    source = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(source, digest_size=16).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2), encoding="utf-8")
    temporary.replace(path)


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


def atomic_json_records(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("[\n")
        for index, record in enumerate(frame.to_dict("records")):
            if index:
                stream.write(",\n")
            json.dump(
                json_safe(record),
                stream,
                ensure_ascii=True,
                separators=(",", ":"),
            )
        stream.write("\n]\n")
    temporary.replace(path)


def build_tracklets(
    annotations: pd.DataFrame,
    samples: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    sample_fields = samples[
        ["token", "scene_token", "sample_index", "timestamp_ns", "raw_frame_index"]
    ].rename(
        columns={
            "token": "sample_token",
            "scene_token": "timeline_scene_token",
            "timestamp_ns": "timeline_timestamp_ns",
            "raw_frame_index": "timeline_raw_frame_index",
        }
    )
    merged = annotations.merge(sample_fields, on="sample_token", how="inner")
    if len(merged) != len(annotations):
        raise ValueError("Some annotations do not resolve to timeline samples")
    if not (merged["scene_token"] == merged["timeline_scene_token"]).all():
        raise ValueError("Annotation and timeline scene tokens disagree")

    rows: list[dict[str, Any]] = []
    for (scene_token, instance_token), group in merged.groupby(
        ["scene_token", "instance_token"], sort=True
    ):
        by_index = {
            int(row.sample_index): row
            for row in group.sort_values("sample_index").itertuples(index=False)
        }
        for start_index in sorted(by_index):
            indices = list(range(start_index, start_index + window_size))
            if not all(index in by_index for index in indices):
                continue
            observations = [by_index[index] for index in indices]
            centres = np.asarray(
                [np.asarray(row.center_world, dtype=np.float64) for row in observations]
            )
            timestamps = np.asarray(
                [int(row.timeline_timestamp_ns) for row in observations], dtype=np.int64
            )
            duration = float((timestamps[-1] - timestamps[0]) / 1e9)
            if duration <= 0:
                continue
            velocity = (centres[-1] - centres[0]) / duration
            speed = float(np.linalg.norm(velocity[:2]))
            window_token = stable_token(
                "tracklet-window-v2", scene_token, start_index, instance_token
            )
            rows.append(
                {
                    "token": window_token,
                    "scene_token": str(scene_token),
                    "log_token": str(observations[0].log_token),
                    "sequence": str(observations[0].sequence),
                    "instance_token": str(instance_token),
                    "semantic_id": int(observations[0].semantic_id),
                    "instance_id": int(observations[0].instance_id),
                    "raw_label": str(observations[0].raw_label),
                    "dynamic_annotation": bool(observations[0].dynamic),
                    "supervised_eligible": bool(observations[0].supervised_eligible),
                    "start_sample_index": start_index,
                    "middle_sample_index": start_index + window_size // 2,
                    "end_sample_index": start_index + window_size - 1,
                    "sample_tokens": [str(row.sample_token) for row in observations],
                    "annotation_tokens": [str(row.token) for row in observations],
                    "raw_frame_indices": [
                        int(row.timeline_raw_frame_index) for row in observations
                    ],
                    "rgb_left_filenames": [
                        str(row.cam0_filename) for row in observations
                    ],
                    "projected_bboxes_xyxy": [
                        row.projected_bbox_xyxy for row in observations
                    ],
                    "timestamps_ns": timestamps.tolist(),
                    "centers_world": centres.reshape(-1).tolist(),
                    "modality_masks": [list(row.modality_mask) for row in observations],
                    "duration_seconds": duration,
                    "velocity_world_mps": velocity.tolist(),
                    "horizontal_speed_mps": speed,
                    "schema_version": SCHEMA_VERSION,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete five-observation tracklets were found")
    if not frame["token"].is_unique:
        raise ValueError("Tracklet tokens are not unique")
    return frame.sort_values(
        ["scene_token", "start_sample_index", "semantic_id", "instance_id"]
    ).reset_index(drop=True)


def temporal_relation(
    subject: Any,
    target: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    centres_a = np.asarray(subject.centers_world, dtype=np.float64).reshape(
        args.window_size, 3
    )
    centres_b = np.asarray(target.centers_world, dtype=np.float64).reshape(
        args.window_size, 3
    )
    relative = centres_a - centres_b
    distances = np.linalg.norm(relative[:, :2], axis=1)
    middle_distance = float(distances[args.window_size // 2])
    if middle_distance > args.candidate_distance:
        return None

    velocity_a = np.asarray(subject.velocity_world_mps, dtype=np.float64)
    velocity_b = np.asarray(target.velocity_world_mps, dtype=np.float64)
    speed_a = float(np.linalg.norm(velocity_a[:2]))
    speed_b = float(np.linalg.norm(velocity_b[:2]))
    distance_steps = np.diff(distances)
    decreasing_fraction = float(np.mean(distance_steps <= -args.step_distance_epsilon))
    increasing_fraction = float(np.mean(distance_steps >= args.step_distance_epsilon))
    distance_change = float(distances[-1] - distances[0])

    direction_cosine = 0.0
    if speed_a >= args.minimum_motion_speed and speed_b >= args.minimum_motion_speed:
        direction_cosine = float(
            np.dot(velocity_a[:2], velocity_b[:2]) / (speed_a * speed_b)
        )
        direction_cosine = float(np.clip(direction_cosine, -1.0, 1.0))

    at_least_one_moving = max(speed_a, speed_b) >= args.minimum_motion_speed
    both_moving = min(speed_a, speed_b) >= args.minimum_motion_speed
    if not at_least_one_moving:
        return None
    path_distance = float("nan")
    if both_moving:
        path_a = LineString(centres_a[:, :2])
        path_b = LineString(centres_b[:, :2])
        path_distance = float(path_a.distance(path_b))

    labels = {
        "approaching": bool(
            at_least_one_moving
            and distance_change <= -args.minimum_distance_change
            and decreasing_fraction >= args.trend_fraction
        ),
        "moving_away": bool(
            at_least_one_moving
            and distance_change >= args.minimum_distance_change
            and increasing_fraction >= args.trend_fraction
        ),
        "same_motion_direction": bool(
            both_moving and direction_cosine >= args.same_direction_cosine
        ),
        "crossing_path": bool(
            both_moving
            and direction_cosine <= args.crossing_max_cosine
            and np.isfinite(path_distance)
            and path_distance <= args.crossing_path_distance
        ),
    }
    vector = [int(labels[name]) for name in TEMPORAL_PREDICATES]
    positives = [name for name in TEMPORAL_PREDICATES if labels[name]]
    row: dict[str, Any] = {
        "token": stable_token("temporal-relation-v2", subject.token, target.token),
        "scene_token": str(subject.scene_token),
        "sequence": str(subject.sequence),
        "start_sample_index": int(subject.start_sample_index),
        "middle_sample_index": int(subject.middle_sample_index),
        "middle_raw_frame_index": int(
            subject.raw_frame_indices[args.window_size // 2]
        ),
        "raw_frame_indices": list(subject.raw_frame_indices),
        "rgb_left_filenames": list(subject.rgb_left_filenames),
        "subject_tracklet_token": str(subject.token),
        "object_tracklet_token": str(target.token),
        "subject_annotation_tokens": list(subject.annotation_tokens),
        "object_annotation_tokens": list(target.annotation_tokens),
        "subject_instance_token": str(subject.instance_token),
        "object_instance_token": str(target.instance_token),
        "subject_label": str(subject.raw_label),
        "object_label": str(target.raw_label),
        "subject_projected_bboxes_xyxy": list(subject.projected_bboxes_xyxy),
        "object_projected_bboxes_xyxy": list(target.projected_bboxes_xyxy),
        "subject_speed_mps": speed_a,
        "object_speed_mps": speed_b,
        "relative_distances_m": distances.tolist(),
        "distance_change_m": distance_change,
        "decreasing_step_fraction": decreasing_fraction,
        "increasing_step_fraction": increasing_fraction,
        "motion_direction_cosine": direction_cosine,
        "observed_path_distance_m": path_distance,
        "middle_distance_m": middle_distance,
        "predicate_vector": vector,
        "positive_predicates": positives,
        "no_relation": not any(vector),
        "supervised_eligible": bool(
            subject.supervised_eligible and target.supervised_eligible
        ),
        "schema_version": SCHEMA_VERSION,
    }
    row.update(labels)
    return row


def build_temporal_relations(
    tracklets: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reverse_labels: dict[tuple[str, str], tuple[bool, ...]] = {}
    for (_, _), group in tracklets.groupby(
        ["scene_token", "start_sample_index"], sort=True
    ):
        records = list(group.itertuples(index=False))
        for subject in records:
            for target in records:
                if subject.instance_token == target.instance_token:
                    continue
                row = temporal_relation(subject, target, args)
                if row is None:
                    continue
                rows.append(row)
                for name in TEMPORAL_PREDICATES:
                    counts[name] += int(row[name])
                reverse_labels[(subject.token, target.token)] = tuple(
                    bool(row[name]) for name in TEMPORAL_PREDICATES
                )

    violations: list[dict[str, Any]] = []
    for (subject, target), labels in reverse_labels.items():
        reverse = reverse_labels.get((target, subject))
        if reverse is None or labels != reverse:
            violations.append({"subject": subject, "object": target})
            if len(violations) >= 20:
                break
    frame = pd.DataFrame(rows, columns=TEMPORAL_RELATION_COLUMNS)
    if not frame["token"].is_unique:
        raise ValueError("Temporal relation tokens are not unique")
    return (
        frame,
        {name: int(counts[name]) for name in TEMPORAL_PREDICATES},
        violations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--predicate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--candidate-distance", type=float, default=30.0)
    parser.add_argument("--minimum-motion-speed", type=float, default=0.50)
    parser.add_argument("--minimum-distance-change", type=float, default=1.0)
    parser.add_argument("--step-distance-epsilon", type=float, default=0.05)
    parser.add_argument("--trend-fraction", type=float, default=0.75)
    parser.add_argument("--same-direction-cosine", type=float, default=0.80)
    parser.add_argument("--crossing-max-cosine", type=float, default=0.50)
    parser.add_argument("--crossing-path-distance", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.window_size != 5:
        parser.error("The frozen V2 protocol requires exactly five observations")

    timeline_root = args.timeline_root.resolve()
    predicate_root = args.predicate_root.resolve()
    output_root = args.output_root.resolve()
    result_file = output_root / "temporal_predicate_build_report.json"
    if result_file.exists() and not args.force:
        raise FileExistsError(f"Output exists: {result_file}. Pass --force to replace.")
    output_root.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(timeline_root / "tables/sample.parquet")
    annotations = pd.read_json(predicate_root / "sample_annotations.json")
    tracklets = build_tracklets(annotations, samples, args.window_size)
    relations, counts, violations = build_temporal_relations(tracklets, args)

    atomic_json_records(tracklets, output_root / "tracklet_windows.json")
    atomic_json_records(relations, output_root / "temporal_relations.json")
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "status": "provisional_candidate_protocol",
        "predicate_order": TEMPORAL_PREDICATES,
        "window_size": args.window_size,
        "thresholds": {
            "candidate_distance": args.candidate_distance,
            "minimum_motion_speed": args.minimum_motion_speed,
            "minimum_distance_change": args.minimum_distance_change,
            "step_distance_epsilon": args.step_distance_epsilon,
            "trend_fraction": args.trend_fraction,
            "same_direction_cosine": args.same_direction_cosine,
            "crossing_max_cosine": args.crossing_max_cosine,
            "crossing_path_distance": args.crossing_path_distance,
        },
    }
    atomic_json(output_root / "temporal_predicate_config_provisional.json", configuration)
    report = {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "tracklet_windows": int(len(tracklets)),
            "distinct_tracklet_instances": int(tracklets["instance_token"].nunique()),
            "dynamic_tracklet_windows": int(tracklets["dynamic_annotation"].sum()),
            "temporal_relation_candidates": int(len(relations)),
            "no_temporal_relation_candidates": int(relations["no_relation"].sum()),
        },
        "predicate_positive_counts": counts,
        "validation": {
            "passed": not violations,
            "unique_tracklet_tokens": bool(tracklets["token"].is_unique),
            "unique_relation_tokens": bool(relations["token"].is_unique),
            "symmetric_relation_violations": violations,
        },
        "warning": (
            "These temporal labels are provisional. Freeze thresholds using training "
            "sequences only after support and trajectory audits."
        ),
    }
    atomic_json(result_file, report)
    print(json.dumps(report, indent=2))
    print(f"Saved temporal predicates to: {output_root}")


if __name__ == "__main__":
    main()
