#!/usr/bin/env python3
"""Validate and summarize per-sequence KITTI-360 timeline outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    sequences = [
        line.strip()
        for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregate_counts = {
        "logs": 0,
        "scenes": 0,
        "samples": 0,
        "raw_frames": 0,
        "sample_data": 0,
        "ego_poses": 0,
        "rejected_segments": 0,
    }
    complete: list[str] = []
    missing: list[str] = []
    failed: dict[str, object] = {}
    reports: dict[str, object] = {}

    for sequence in sequences:
        report_path = output_root / "parts" / sequence / "scene_build_report.json"
        if not report_path.exists():
            missing.append(sequence)
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports[sequence] = {
            "counts": report.get("counts", {}),
            "sequence": report.get("sequences", {}).get(sequence, {}),
            "validation": report.get("validation", {}),
        }
        validation = report.get("validation", {})
        if not validation.get("passed", False):
            failed[sequence] = validation
            continue
        complete.append(sequence)
        for key in aggregate_counts:
            aggregate_counts[key] += int(report.get("counts", {}).get(key, 0))

    result = {
        "layout": "partitioned_by_original_kitti360_sequence",
        "output_root": str(output_root),
        "expected_sequences": sequences,
        "complete_sequences": complete,
        "missing_sequences": missing,
        "failed_sequences": failed,
        "counts": aggregate_counts,
        "sequence_reports": reports,
        "validation": {
            "passed": len(complete) == len(sequences) and not missing and not failed,
            "complete_count": len(complete),
            "expected_count": len(sequences),
        },
    }
    target = output_root / "full_dataset_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps(result["counts"], indent=2))
    print(json.dumps(result["validation"], indent=2))
    if missing:
        print("Missing:", ", ".join(missing))
    if failed:
        print("Failed:", ", ".join(failed))
    print(f"Saved: {target}")
    if not result["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
