from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = "IBP-K360-saved-triplet-ranking-v1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def ranked_triplet_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    group_ids: list[str],
    predicate_names: tuple[str, ...],
    eligible_rows: np.ndarray | None = None,
    ks: tuple[int, ...] = (10, 20, 50, 100),
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.uint8)
    if logits.shape != targets.shape:
        raise ValueError("Logits and targets must have identical shapes")
    if logits.ndim != 2 or logits.shape[1] != len(predicate_names):
        raise ValueError("Prediction shape does not match the predicate vocabulary")
    if len(group_ids) != len(logits):
        raise ValueError("Group IDs and prediction rows are misaligned")
    if eligible_rows is None:
        eligible = np.ones(len(logits), dtype=np.bool_)
    else:
        eligible = np.asarray(eligible_rows, dtype=np.bool_)
        if eligible.shape != (len(logits),):
            raise ValueError("Eligibility mask must contain one value per relation row")

    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)

    support = targets.sum(axis=0, dtype=np.int64)
    total_positive = int(support.sum())
    eligible_positive = int(targets[eligible].sum())
    result: dict[str, Any] = {
        "definition": (
            "Predicate triplets are ranked within each frame or temporal clip. "
            "Unavailable predicted tracklets remain in the ground-truth denominator "
            "but cannot enter the ranked candidate list."
        ),
        "examples": int(len(targets)),
        "eligible_examples": int(eligible.sum()),
        "candidate_coverage": float(eligible.mean()) if len(eligible) else 0.0,
        "positive_triplets": total_positive,
        "eligible_positive_triplets": eligible_positive,
    }

    for k in ks:
        recovered_by_class = np.zeros(len(predicate_names), dtype=np.int64)
        for indices in grouped.values():
            rows = np.asarray(indices, dtype=np.int64)
            rows = rows[eligible[rows]]
            if not len(rows):
                continue
            local_scores = scores[rows].reshape(-1)
            keep = min(k, len(local_scores))
            selected = np.argsort(-local_scores, kind="mergesort")[:keep]
            relation_indices = selected // len(predicate_names)
            predicate_indices = selected % len(predicate_names)
            selected_targets = targets[
                rows[relation_indices], predicate_indices
            ].astype(np.bool_)
            for predicate_index in predicate_indices[selected_targets]:
                recovered_by_class[int(predicate_index)] += 1

        per_predicate = {
            name: recovered_by_class[index] / max(int(support[index]), 1)
            for index, name in enumerate(predicate_names)
        }
        supported_recalls = [
            per_predicate[name]
            for index, name in enumerate(predicate_names)
            if support[index] > 0
        ]
        result[f"R@{k}"] = int(recovered_by_class.sum()) / max(total_positive, 1)
        result[f"mR@{k}"] = (
            float(np.mean(supported_recalls)) if supported_recalls else 0.0
        )
        result[f"per_predicate_R@{k}"] = per_predicate
    return result


def read_predicate_names(group: h5py.Group) -> tuple[str, ...]:
    raw = group.attrs.get("predicate_names")
    if raw is None:
        raise ValueError(f"Missing predicate_names attribute in {group.name}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    names = tuple(str(value) for value in json.loads(str(raw)))
    if not names:
        raise ValueError(f"Empty predicate vocabulary in {group.name}")
    return names


def evaluate_saved_predictions(prediction_path: Path) -> dict[str, Any]:
    prediction_path = prediction_path.resolve()
    with h5py.File(prediction_path, "r") as predictions:
        output: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "split": "held_out_test",
            "source_predictions": str(prediction_path),
            "source_predictions_sha256": sha256_file(prediction_path),
        }
        for task in ("spatial", "temporal"):
            if task not in predictions:
                raise ValueError(f"Missing prediction group: {task}")
            group = predictions[task]
            required = {"raw_logits", "targets", "group_ids"}
            missing = sorted(required - set(group))
            if missing:
                raise ValueError(f"Missing datasets in {group.name}: {missing}")
            availability = None
            if task == "temporal":
                if "tracklet_available" not in group:
                    raise ValueError("Temporal predictions lack tracklet_available")
                availability = np.asarray(group["tracklet_available"], dtype=np.bool_)
            output[task] = ranked_triplet_metrics(
                np.asarray(group["raw_logits"], dtype=np.float32),
                np.asarray(group["targets"], dtype=np.uint8),
                decode_strings(np.asarray(group["group_ids"])),
                read_predicate_names(group),
                eligible_rows=availability,
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute end-to-end R@K and mR@K from frozen raw predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}")
    result = evaluate_saved_predictions(args.predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
