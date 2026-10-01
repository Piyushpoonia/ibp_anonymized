#!/usr/bin/env python3
"""Validate one generated spatial/temporal KITTI-360 predicate partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPATIAL_SCHEMA = "IBP-K360-predicates-v3.0.0-canonical-candidate"
TEMPORAL_SCHEMA = "IBP-K360-temporal-predicates-v2.2.0-candidate"
SPATIAL_PREDICATES = [
    "left_of",
    "in_front_of",
    "near",
    "overlapping",
    "occluding",
]


def first_json_record(path: Path) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8") as stream:
        decoder = json.JSONDecoder()
        text = ""
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return None
            text += chunk
            start = text.find("[")
            if start < 0:
                continue
            remainder = text[start + 1 :].lstrip()
            if remainder.startswith("]"):
                return None
            try:
                value, _ = decoder.raw_decode(remainder)
                return value
            except json.JSONDecodeError:
                continue


def require_files(root: Path, names: list[str], errors: list[str]) -> None:
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{path}")


def validate_spatial(root: Path, errors: list[str]) -> None:
    require_files(
        root,
        [
            "predicate_build_report.json",
            "spatial_relations.json",
            "sample_annotations.json",
            "categories.json",
            "instances.json",
            "predicate_config_provisional.json",
            "predicate_statistics.json",
        ],
        errors,
    )
    report_path = root / "predicate_build_report.json"
    relation_path = root / "spatial_relations.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != SPATIAL_SCHEMA:
            errors.append(f"wrong_spatial_schema:{report.get('schema_version')}")
        if not report.get("validation", {}).get("passed", False):
            errors.append("spatial_validation_failed")
        if int(report.get("counts", {}).get("relation_candidates", 0)) <= 0:
            errors.append("no_spatial_relation_candidates")
        if list(report.get("predicate_positive_counts", {}).keys()) != SPATIAL_PREDICATES:
            errors.append("wrong_spatial_predicate_vocabulary")
    config_path = root / "predicate_config_provisional.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("predicate_order") != SPATIAL_PREDICATES:
            errors.append("wrong_spatial_predicate_order")
        if not config.get("canonical_inverse_encoding"):
            errors.append("missing_canonical_inverse_encoding")
    if relation_path.is_file() and relation_path.stat().st_size:
        record = first_json_record(relation_path)
        required = {
            "raw_frame_index",
            "rgb_left_filename",
            "subject_projected_bbox_xyxy",
            "object_projected_bbox_xyxy",
            "positive_predicates",
            "predicate_vector",
            "occlusion_mask_supported",
        }
        if record is None or not required.issubset(record):
            errors.append("spatial_relation_fields_missing")
        elif (
            len(record.get("predicate_vector", [])) != len(SPATIAL_PREDICATES)
            or not set(record.get("positive_predicates", [])).issubset(SPATIAL_PREDICATES)
        ):
            errors.append("invalid_spatial_predicate_encoding")


def validate_temporal(root: Path, errors: list[str]) -> None:
    temporal = root / "temporal"
    require_files(
        temporal,
        [
            "temporal_predicate_build_report.json",
            "temporal_relations.json",
            "tracklet_windows.json",
            "temporal_predicate_config_provisional.json",
        ],
        errors,
    )
    report_path = temporal / "temporal_predicate_build_report.json"
    relation_path = temporal / "temporal_relations.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != TEMPORAL_SCHEMA:
            errors.append(f"wrong_temporal_schema:{report.get('schema_version')}")
        if not report.get("validation", {}).get("passed", False):
            errors.append("temporal_validation_failed")
        if int(report.get("counts", {}).get("tracklet_windows", 0)) <= 0:
            errors.append("no_tracklet_windows")
    if relation_path.is_file() and relation_path.stat().st_size:
        record = first_json_record(relation_path)
        required = {
            "raw_frame_indices",
            "middle_raw_frame_index",
            "rgb_left_filenames",
            "subject_projected_bboxes_xyxy",
            "object_projected_bboxes_xyxy",
            "subject_radial_velocity_toward_object_mps",
            "distance_change_m",
            "decreasing_step_fraction",
            "increasing_step_fraction",
            "joint_visible_steps",
            "human_review_eligible",
            "subject_motion_toward_object",
            "subject_motion_away_from_object",
            "pair_distance_decreasing",
            "pair_distance_increasing",
            "positive_predicates",
        }
        if record is None or not required.issubset(record):
            errors.append("temporal_relation_fields_missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("spatial", "temporal", "both"), default="both")
    args = parser.parse_args()

    root = args.part_root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"missing_part_root:{root}")
    else:
        if args.mode in {"spatial", "both"}:
            validate_spatial(root, errors)
        if args.mode in {"temporal", "both"}:
            validate_temporal(root, errors)
        temporary_files = [str(path) for path in root.rglob("*.tmp")]
        if temporary_files:
            errors.extend(f"unfinished_temporary:{path}" for path in temporary_files)

    result = {
        "part_root": str(root),
        "mode": args.mode,
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
