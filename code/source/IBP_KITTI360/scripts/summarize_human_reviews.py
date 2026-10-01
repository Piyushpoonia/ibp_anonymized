#!/usr/bin/env python3
"""Summarize primary and common-overlap human predicate review progress."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SPATIAL_PREDICATES = {"left_of", "in_front_of", "near", "overlapping", "occluding"}
TEMPORAL_PREDICATES = {
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
}


def load_records(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def status(records: list[dict[str, object]], allowed: set[str]) -> dict[str, object]:
    decisions = Counter(
        "pending" if row.get("human_valid") is None else str(bool(row["human_valid"])).lower()
        for row in records
    )
    by_predicate: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_corrections: list[dict[str, object]] = []
    for row in records:
        decision = (
            "pending" if row.get("human_valid") is None else str(bool(row["human_valid"])).lower()
        )
        by_predicate[str(row.get("reviewed_as", "unknown"))][decision] += 1
        corrections = row.get("human_corrected_predicates", [])
        if not isinstance(corrections, list) or not set(corrections).issubset(allowed):
            invalid_corrections.append(
                {
                    "relation_token": row.get("relation_token"),
                    "human_corrected_predicates": corrections,
                }
            )
    return {
        "records": len(records),
        "decisions": dict(decisions),
        "by_predicate": {name: dict(counts) for name, counts in sorted(by_predicate.items())},
        "invalid_corrections": invalid_corrections,
    }


def common_agreement(common_root: Path) -> dict[str, object]:
    decisions: dict[tuple[str, str], dict[str, object]] = defaultdict(dict)
    for reviewer_dir in sorted(common_root.glob("member_*")):
        for kind in ("spatial", "temporal"):
            path = reviewer_dir / f"{kind}_review.json"
            if not path.is_file():
                continue
            for row in load_records(path):
                key = (kind, str(row["relation_token"]))
                decisions[key][reviewer_dir.name] = row.get("human_valid")

    complete = 0
    unanimous = 0
    disagreements: list[dict[str, object]] = []
    for (kind, token), reviewer_decisions in sorted(decisions.items()):
        values = list(reviewer_decisions.values())
        if len(values) == 3 and all(value is not None for value in values):
            complete += 1
            if len(set(values)) == 1:
                unanimous += 1
            else:
                disagreements.append(
                    {"kind": kind, "relation_token": token, "decisions": reviewer_decisions}
                )
    return {
        "items_found": len(decisions),
        "items_reviewed_by_all_three": complete,
        "unanimous_items": unanimous,
        "unanimous_percent": 100.0 * unanimous / complete if complete else None,
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.reviews_root.resolve()
    reviewers: dict[str, object] = {}
    packages_found = 0
    for reviewer_dir in sorted(root.glob("member_*")):
        package_results: dict[str, object] = {}
        for sequence_dir in sorted(path for path in reviewer_dir.iterdir() if path.is_dir()):
            spatial_path = sequence_dir / "spatial_review.json"
            temporal_path = sequence_dir / "temporal_review.json"
            if not spatial_path.is_file() or not temporal_path.is_file():
                continue
            packages_found += 1
            package_results[sequence_dir.name] = {
                "spatial": status(load_records(spatial_path), SPATIAL_PREDICATES),
                "temporal": status(load_records(temporal_path), TEMPORAL_PREDICATES),
            }
        reviewers[reviewer_dir.name] = package_results

    result = {
        "expected_primary_packages": 9,
        "primary_packages_found": packages_found,
        "all_primary_packages_present": packages_found == 9,
        "predicate_count": 9,
        "spatial_predicates": sorted(SPATIAL_PREDICATES),
        "temporal_predicates": sorted(TEMPORAL_PREDICATES),
        "reviewers": reviewers,
        "common_overlap": common_agreement(root / "common_overlap"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
