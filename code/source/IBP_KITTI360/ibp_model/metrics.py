from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order].astype(np.float64)
    precision = np.cumsum(sorted_target) / np.arange(1, len(target) + 1)
    return float((precision * sorted_target).sum() / positives)


def multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    predicate_names: tuple[str, ...],
    threshold: float = 0.5,
) -> dict[str, Any]:
    probabilities = torch.sigmoid(logits).cpu().numpy()
    targets = labels.cpu().numpy().astype(np.uint8)
    predictions = probabilities >= threshold
    per_predicate: dict[str, dict[str, float | int | None]] = {}
    supported_f1: list[float] = []
    supported_ap: list[float] = []
    unsupported: list[str] = []
    for index, name in enumerate(predicate_names):
        target = targets[:, index].astype(bool)
        predicted = predictions[:, index]
        tp = int(np.logical_and(target, predicted).sum())
        fp = int(np.logical_and(~target, predicted).sum())
        fn = int(np.logical_and(target, ~predicted).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        ap = average_precision(targets[:, index], probabilities[:, index])
        if target.any():
            supported_f1.append(f1)
        else:
            unsupported.append(name)
        if math.isfinite(ap):
            supported_ap.append(ap)
        per_predicate[name] = {
            "support": int(target.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": ap if math.isfinite(ap) else None,
        }

    flat_target = targets.astype(bool).reshape(-1)
    flat_prediction = predictions.reshape(-1)
    tp = int(np.logical_and(flat_target, flat_prediction).sum())
    fp = int(np.logical_and(~flat_target, flat_prediction).sum())
    fn = int(np.logical_and(flat_target, ~flat_prediction).sum())
    micro_precision = tp / max(tp + fp, 1)
    micro_recall = tp / max(tp + fn, 1)
    micro_f1 = 2.0 * micro_precision * micro_recall / max(
        micro_precision + micro_recall, 1e-12
    )
    return {
        "threshold": threshold,
        "examples": int(len(targets)),
        "macro_f1": float(np.mean(supported_f1)) if supported_f1 else None,
        "mean_average_precision": float(np.mean(supported_ap)) if supported_ap else None,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_match_accuracy": float(np.all(targets == predictions, axis=1).mean()),
        "unsupported_predicates": unsupported,
        "per_predicate": per_predicate,
    }
