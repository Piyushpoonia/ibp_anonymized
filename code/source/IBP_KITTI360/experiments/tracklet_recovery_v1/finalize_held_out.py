from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...ibp_model.predicted_tracklets import sha256_file
from . import RECOVERY_SCHEMA_VERSION
from .apply_frozen_held_out import (
    EXPECTED_TEST_SEQUENCES,
    FROZEN_MARGIN,
    FROZEN_MODE,
    FROZEN_VARIANT,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = args.metrics.resolve()
    manifest_path = args.recovery_manifest.resolve()
    metrics = read_json(metrics_path)
    manifest = read_json(manifest_path)
    if metrics.get("split") != "held_out_test" or not metrics.get("publishable_final_result"):
        raise ValueError("Expected completed held-out Stage-3 metrics")
    if manifest.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise ValueError("Unexpected recovery manifest schema")
    if manifest.get("frozen_variant") != FROZEN_VARIANT:
        raise ValueError("Held-out tracks did not use the frozen variant")
    if manifest.get("frozen_mode") != FROZEN_MODE:
        raise ValueError("Held-out tracks did not use the frozen mode")
    if float(manifest.get("frozen_margin_threshold", -1.0)) != FROZEN_MARGIN:
        raise ValueError("Held-out tracks did not use the frozen margin")
    if set(metrics.get("test_sequences", [])) != EXPECTED_TEST_SEQUENCES:
        raise ValueError("Metrics do not contain exactly held-out sequences 0007 and 0009")

    manifest_sha256 = sha256_file(manifest_path)
    existing = metrics.get("tracklet_recovery")
    if existing is not None:
        if existing.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Metrics were already finalized with a different recovery manifest")
        return metrics

    reports = manifest.get("reports", [])
    keys = (
        "original_mutual_links",
        "fallback_links",
        "correct_fallback_links",
        "predicted_recovery_links",
        "correct_recovery_links",
        "true_forward_links",
    )
    totals = {key: sum(int(report.get(key, 0)) for report in reports) for key in keys}
    totals["recovery_link_precision"] = totals["correct_recovery_links"] / max(
        totals["predicted_recovery_links"], 1
    )
    totals["recovery_link_recall"] = totals["correct_recovery_links"] / max(
        totals["true_forward_links"], 1
    )
    metrics["tracklet_recovery"] = {
        "method": "validation-selected one-to-one fallback association recovery",
        "variant": FROZEN_VARIANT,
        "mode": FROZEN_MODE,
        "dustbin_margin_threshold": FROZEN_MARGIN,
        "selected_on_sequence": manifest["validation_sequence"],
        "ground_truth_used_for_recovery": False,
        "ground_truth_used_for_post_hoc_metrics": True,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        **totals,
    }
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temporary.replace(metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach frozen recovery provenance to final metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(finalize(parse_args()), indent=2))


if __name__ == "__main__":
    main()
