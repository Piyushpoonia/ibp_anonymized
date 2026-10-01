#!/usr/bin/env python3
"""Aggregate validation reports from KITTI-360 predicate partitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SPATIAL_SCHEMA = "IBP-K360-predicates-v3.0.0-canonical-candidate"
TEMPORAL_SCHEMA = "IBP-K360-temporal-predicates-v2.2.0-candidate"
SPATIAL_PREDICATES = ["left_of", "in_front_of", "near", "overlapping", "occluding"]
TEMPORAL_PREDICATES = [
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
]


def read_sequences(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    args = parser.parse_args()

    expected_sequences = read_sequences(args.sequence_file)
    expected_set = set(expected_sequences)
    reports = sorted(args.parts_root.glob("*/predicate_build_report.json"))
    if not reports:
        raise FileNotFoundError(f"No predicate reports found below {args.parts_root}")

    totals: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    temporal_totals: Counter[str] = Counter()
    temporal_predicate_counts: Counter[str] = Counter()
    sequences: dict[str, object] = {}
    all_passed = True
    spatial_schema_versions: set[str] = set()
    temporal_schema_versions: set[str] = set()
    found_sequences: set[str] = set()
    temporal_sequences: set[str] = set()
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sequence = str(report["sequence"])
        found_sequences.add(sequence)
        spatial_schema_versions.add(str(report["schema_version"]))
        totals.update({key: int(value) for key, value in report["counts"].items()})
        predicate_counts.update(
            {key: int(value) for key, value in report["predicate_positive_counts"].items()}
        )
        passed = bool(report["validation"]["passed"])
        all_passed = all_passed and passed
        sequence_result: dict[str, object] = {
            "counts": report["counts"],
            "predicate_positive_counts": report["predicate_positive_counts"],
            "validation_passed": passed,
        }
        temporal_report_path = report_path.parent / "temporal" / "temporal_predicate_build_report.json"
        if temporal_report_path.exists():
            temporal = json.loads(temporal_report_path.read_text(encoding="utf-8"))
            temporal_sequences.add(sequence)
            temporal_schema_versions.add(str(temporal["schema_version"]))
            temporal_totals.update(
                {key: int(value) for key, value in temporal["counts"].items()}
            )
            temporal_predicate_counts.update(
                {
                    key: int(value)
                    for key, value in temporal["predicate_positive_counts"].items()
                }
            )
            temporal_passed = bool(temporal["validation"]["passed"])
            all_passed = all_passed and temporal_passed
            sequence_result["temporal"] = {
                "counts": temporal["counts"],
                "predicate_positive_counts": temporal["predicate_positive_counts"],
                "validation_passed": temporal_passed,
            }
        sequences[sequence] = sequence_result

    missing_sequences = sorted(expected_set - found_sequences)
    missing_temporal_sequences = sorted(expected_set - temporal_sequences)
    unexpected_sequences = sorted(found_sequences - expected_set)
    schemas_passed = (
        spatial_schema_versions == {SPATIAL_SCHEMA}
        and temporal_schema_versions == {TEMPORAL_SCHEMA}
    )
    all_passed = (
        all_passed
        and not missing_sequences
        and not missing_temporal_sequences
        and not unexpected_sequences
        and len(found_sequences) == len(expected_sequences)
        and schemas_passed
    )

    relation_count = totals["relation_candidates"]
    result = {
        "candidate_only": True,
        "predicate_count": 9,
        "spatial_predicates": SPATIAL_PREDICATES,
        "temporal_predicates": TEMPORAL_PREDICATES,
        "expected_partitions": len(expected_sequences),
        "partitions_found": len(reports),
        "all_partitions_passed": all_passed,
        "spatial_schema_versions": sorted(spatial_schema_versions),
        "temporal_schema_versions": sorted(temporal_schema_versions),
        "schema_check_passed": schemas_passed,
        "missing_sequences": missing_sequences,
        "missing_temporal_sequences": missing_temporal_sequences,
        "unexpected_sequences": unexpected_sequences,
        "totals": dict(totals),
        "predicate_positive_counts": dict(predicate_counts),
        "predicate_positive_rate_percent": {
            name: 100.0 * count / relation_count if relation_count else 0.0
            for name, count in predicate_counts.items()
        },
        "temporal_totals": dict(temporal_totals),
        "temporal_predicate_positive_counts": dict(temporal_predicate_counts),
        "sequences": sequences,
        "warning": (
            "Freeze vocabulary and thresholds from training sequences only before training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["totals"], indent=2))
    print(json.dumps(result["predicate_positive_counts"], indent=2))
    print(json.dumps(result["temporal_totals"], indent=2))
    print(json.dumps(result["temporal_predicate_positive_counts"], indent=2))
    print(f"All partitions passed: {all_passed}")
    print(f"Saved: {args.output}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
