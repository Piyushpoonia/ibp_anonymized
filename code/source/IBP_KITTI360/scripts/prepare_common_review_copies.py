#!/usr/bin/env python3
"""Copy one common review template to independent reviewer directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def update_identity(path: Path, reviewer: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for record in value:
            record["reviewer_id"] = reviewer
            record["review_set"] = "common_overlap"
    else:
        value["reviewer_id"] = reviewer
        value["review_set"] = "common_overlap"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reviewers", nargs="+", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    template = args.template.resolve()
    output_root = args.output_root.resolve()
    if not (template / "review_summary.json").is_file():
        raise FileNotFoundError(f"Incomplete common review template: {template}")
    output_root.mkdir(parents=True, exist_ok=True)

    for reviewer in args.reviewers:
        target = (output_root / reviewer).resolve()
        if target.parent != output_root:
            raise ValueError(f"Unsafe reviewer directory: {target}")
        if target.exists():
            if not args.force:
                print(f"Preserving existing common review: {target}")
                continue
            shutil.rmtree(target)
        shutil.copytree(template, target)
        for name in ("spatial_review.json", "temporal_review.json", "review_summary.json"):
            update_identity(target / name, reviewer)
        print(f"Prepared common review for {reviewer}: {target}")


if __name__ == "__main__":
    main()
