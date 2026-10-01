from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py

from ...ibp_model.predicted_tracklets import sha256_file
from . import RECOVERY_SCHEMA_VERSION
from .recover import recover_track_file


FROZEN_VARIANT = "all_margin_0p00"
FROZEN_MODE = "all"
FROZEN_MARGIN = 0.0
EXPECTED_VALIDATION_SEQUENCE = "2013_05_28_drive_0006_sync"
EXPECTED_TEST_SEQUENCES = {
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0009_sync",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def validate_frozen_selection(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise ValueError("Unexpected validation-recovery schema")
    if summary.get("selection_split") != "validation":
        raise ValueError("Recovery configuration was not selected on validation")
    if summary.get("validation_sequence") != EXPECTED_VALIDATION_SEQUENCE:
        raise ValueError("Recovery configuration used an unexpected validation sequence")
    selected = summary.get("selected_variant", {})
    expected = (FROZEN_VARIANT, FROZEN_MODE, FROZEN_MARGIN)
    actual = (
        selected.get("variant"),
        selected.get("mode"),
        float(selected.get("margin_threshold", float("nan"))),
    )
    if actual != expected:
        raise ValueError(f"Frozen recovery selection differs: expected {expected}, got {actual}")


def existing_report(output: Path, source: Path) -> dict[str, Any] | None:
    report_path = output.with_suffix(".report.json")
    if not output.is_file() or not report_path.is_file():
        return None
    report = read_json(report_path)
    with h5py.File(output, "r", swmr=True) as recovered:
        valid = (
            str(recovered.attrs.get("recovery_mode", "")) == FROZEN_MODE
            and float(recovered.attrs.get("recovery_margin_threshold", -1.0)) == FROZEN_MARGIN
            and str(recovered.attrs.get("base_track_sha256", "")) == sha256_file(source)
        )
    if not valid or "predicted_recovery_links" not in report:
        raise ValueError(f"Existing recovered track is stale or incomplete: {output}")
    return report


def apply(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.validation_summary)
    validate_frozen_selection(summary)
    source_by_sequence = {path.stem: path.resolve() for path in args.source_tracks}
    relation_by_sequence = {path.stem: path.resolve() for path in args.relation_files}
    if set(source_by_sequence) != EXPECTED_TEST_SEQUENCES:
        raise ValueError("Source tracks must contain exactly held-out sequences 0007 and 0009")
    if set(relation_by_sequence) != EXPECTED_TEST_SEQUENCES:
        raise ValueError("Relation files must contain exactly held-out sequences 0007 and 0009")

    output_root = args.output_root.resolve()
    manifest_path = output_root / "held_out_recovery_manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = read_json(manifest_path)
        if manifest.get("validation_summary_sha256") != sha256_file(args.validation_summary):
            raise ValueError("Existing held-out manifest used a different validation selection")
        return manifest

    reports: list[dict[str, Any]] = []
    for sequence in sorted(EXPECTED_TEST_SEQUENCES):
        source = source_by_sequence[sequence]
        relation = relation_by_sequence[sequence]
        output = output_root / "tracks" / FROZEN_VARIANT / f"{sequence}.h5"
        report = None if args.force else existing_report(output, source)
        if report is None:
            report = recover_track_file(
                source,
                output,
                FROZEN_MODE,
                FROZEN_MARGIN,
                args.force,
                association_relation_path=relation,
            )
        reports.append(report)

    manifest = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "selection_split": "validation",
        "validation_sequence": EXPECTED_VALIDATION_SEQUENCE,
        "validation_summary": str(args.validation_summary.resolve()),
        "validation_summary_sha256": sha256_file(args.validation_summary),
        "frozen_variant": FROZEN_VARIANT,
        "frozen_mode": FROZEN_MODE,
        "frozen_margin_threshold": FROZEN_MARGIN,
        "held_out_test_opened": True,
        "held_out_sequences": sorted(EXPECTED_TEST_SEQUENCES),
        "ground_truth_used_for_recovery": False,
        "ground_truth_used_for_post_hoc_association_metrics": True,
        "reports": reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the validation-frozen tracklet recovery rule once to held-out data."
    )
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--source-tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--relation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(apply(parse_args()), indent=2))


if __name__ == "__main__":
    main()
