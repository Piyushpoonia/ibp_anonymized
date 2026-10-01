#!/usr/bin/env python3
"""Validate and summarize predicate-folder human correction files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


SPATIAL = {"left_of", "in_front_of", "near", "overlapping", "occluding"}
TEMPORAL = {
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
}
DECISIONS = {None, "correct", "incorrect", "ambiguous"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.reviews_root.resolve()
    decision_counts: Counter[str] = Counter()
    folder_counts: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, object]] = []
    relation_decisions: dict[str, set[str]] = defaultdict(set)
    files = sorted(root.glob("*/**/corrections.json"))
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            errors.append({"file": str(path), "error": "not_a_json_list"})
            continue
        for record in records:
            relation_id = str(record.get("relation_id", ""))
            kind = str(record.get("kind", ""))
            decision = record.get("human_decision")
            decision_name = "pending" if decision is None else str(decision)
            allowed = SPATIAL if kind == "spatial" else TEMPORAL if kind == "temporal" else set()
            corrected = record.get("corrected_predicates", [])
            decision_counts[decision_name] += 1
            folder_counts[path.parent.name][decision_name] += 1
            relation_decisions[relation_id].add(
                json.dumps([decision, corrected], sort_keys=True)
            )
            if not relation_id:
                errors.append({"file": str(path), "error": "missing_relation_id"})
            if decision not in DECISIONS:
                errors.append(
                    {"relation_id": relation_id, "error": "invalid_human_decision"}
                )
            if not isinstance(corrected, list) or not set(corrected).issubset(allowed):
                errors.append(
                    {
                        "relation_id": relation_id,
                        "error": "invalid_corrected_predicates",
                        "value": corrected,
                    }
                )
            if decision == "incorrect" and not isinstance(corrected, list):
                errors.append(
                    {"relation_id": relation_id, "error": "incorrect_without_correction_list"}
                )

    inconsistent = sorted(
        relation_id
        for relation_id, values in relation_decisions.items()
        if relation_id and len(values) > 1
    )
    result = {
        "reviews_root": str(root),
        "correction_files": len(files),
        "records": sum(decision_counts.values()),
        "decision_counts": dict(decision_counts),
        "by_folder": {
            name: dict(counts) for name, counts in sorted(folder_counts.items())
        },
        "inconsistent_duplicate_relation_ids": inconsistent,
        "validation_passed": not errors and not inconsistent,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors or inconsistent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
