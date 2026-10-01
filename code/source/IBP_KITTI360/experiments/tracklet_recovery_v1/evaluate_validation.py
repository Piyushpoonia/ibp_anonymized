from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ...ibp_model.evaluate_stage3 import relation_metadata
from ...ibp_model.analysis.tracklet_quality_v1.analyze import analyze
from ...ibp_model.analysis.tracklet_quality_v1.evaluate_validation import (
    predicted_predictions,
    write_predictions,
)
from . import RECOVERY_SCHEMA_VERSION


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def run_audit(
    teacher_predictions: Path,
    predicted_predictions_path: Path,
    feature_metadata: Path,
    relation_file: Path,
    track_file: Path,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    metrics_path = output_root / "tracklet_quality_metrics.json"
    if metrics_path.exists() and not force:
        return read_json(metrics_path)
    namespace = SimpleNamespace(
        teacher_predictions=teacher_predictions,
        predicted_predictions=predicted_predictions_path,
        feature_metadata=[feature_metadata],
        relation_files=[relation_file],
        track_files=[track_file],
        output_root=output_root,
        split_name="validation",
        force=force,
    )
    return analyze(namespace)


def result_row(
    name: str,
    track_file: Path,
    audit: dict[str, Any],
    recovery: dict[str, Any] | None,
) -> dict[str, Any]:
    overall = audit["overall"]
    available = audit["available_only"]
    predicted = overall["predicted_tracklet"]
    available_predicted = available["predicted_tracklet"]
    return {
        "variant": name,
        "mode": "mutual" if recovery is None else recovery["mode"],
        "margin_threshold": None if recovery is None else recovery["margin_threshold"],
        "fallback_links": 0 if recovery is None else recovery["fallback_links"],
        "track_file": str(track_file.resolve()),
        "examples": overall["examples"],
        "coverage": overall["tracklet_availability"],
        "identity_error_rate": overall["identity_error_rate"],
        "temporal_macro_f1": predicted["macro_f1"],
        "temporal_map": predicted["mean_average_precision"],
        "temporal_micro_f1": predicted["micro_f1"],
        "exact_match_accuracy": predicted["exact_match_accuracy"],
        "available_only_macro_f1": available_predicted["macro_f1"],
        "available_only_map": available_predicted["mean_average_precision"],
        "perfect_5": audit["quality_counts"].get("perfect_5", 0),
        "good_4": audit["quality_counts"].get("good_4", 0),
        "partial_3": audit["quality_counts"].get("partial_3", 0),
        "poor_identity": audit["quality_counts"].get("poor_identity", 0),
        "missing": audit["quality_counts"].get("missing", 0),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["temporal_macro_f1"] or -1.0),
        float(row["temporal_map"] or -1.0),
        -float(row["identity_error_rate"] or 0.0),
        float(row["coverage"] or 0.0),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    summary_path = output_root / "validation_recovery_summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"Recovery summary already exists: {summary_path}")
    manifest = read_json(args.recovery_manifest)
    if manifest.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise ValueError("Unexpected recovery manifest schema")
    if manifest.get("selection_split") != "validation":
        raise ValueError("Recovery variants were not built for validation selection")

    sequence = args.relation_file.stem
    device = torch.device(args.device)
    metadata = relation_metadata([args.relation_file], "temporal")
    rows: list[dict[str, Any]] = []
    baseline_audit = run_audit(
        args.teacher_predictions,
        args.baseline_predictions,
        args.feature_metadata,
        args.relation_file,
        args.baseline_track,
        output_root / "baseline_mutual" / "audit",
        args.force,
    )
    rows.append(
        result_row("baseline_mutual", args.baseline_track, baseline_audit, None)
    )

    for variant in manifest["variants"]:
        name = str(variant["name"])
        track_file = Path(variant["output"])
        variant_root = output_root / name
        prediction_path = variant_root / "predicted_raw_predictions.h5"
        if not prediction_path.exists() or args.force:
            logits, targets, extras, _, _ = predicted_predictions(
                [args.feature_file],
                [args.relation_file],
                [track_file],
                args.stage3_checkpoint,
                args.batch_size,
                args.workers,
                device,
            )
            variant_root.mkdir(parents=True, exist_ok=True)
            write_predictions(
                prediction_path,
                logits,
                targets,
                metadata,
                args.stage3_checkpoint,
                [sequence],
                **extras,
            )
        audit = run_audit(
            args.teacher_predictions,
            prediction_path,
            args.feature_metadata,
            args.relation_file,
            track_file,
            variant_root / "audit",
            args.force,
        )
        rows.append(result_row(name, track_file, audit, variant))

    selected = max(rows, key=selection_key)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "validation_recovery_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "selection_split": "validation",
        "validation_sequence": sequence,
        "held_out_test_opened_by_this_experiment": False,
        "selection_metric": (
            "lexicographic temporal macro-F1, temporal mAP, lower identity-error rate, coverage"
        ),
        "selected_variant": selected,
        "baseline": rows[0],
        "variants": rows,
        "next_step": (
            "Freeze this result before deciding whether the selected recovery rule merits "
            "a separately named held-out evaluation."
        ),
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tracklet-recovery variants on frozen validation sequence 0006."
    )
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--relation-file", type=Path, required=True)
    parser.add_argument("--baseline-track", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
