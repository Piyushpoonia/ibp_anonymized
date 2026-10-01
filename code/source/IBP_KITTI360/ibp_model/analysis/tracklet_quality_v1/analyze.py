from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from . import QUALITY_BIN_ORDER


SCHEMA_VERSION = "IBP-K360-tracklet-quality-analysis-v1.0.0"
PREDICATES = ("approaching", "moving_away", "same_motion_direction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_array(dataset: h5py.Dataset) -> list[str]:
    return [str(value) for value in dataset.asstr()[:].tolist()]


@dataclass(frozen=True)
class PredictionBundle:
    relation_ids: list[str]
    sequences: list[str]
    logits: np.ndarray
    targets: np.ndarray
    extras: dict[str, np.ndarray]


def load_predictions(path: Path) -> PredictionBundle:
    with h5py.File(path.resolve(), "r", swmr=True) as handle:
        if "temporal" not in handle:
            raise ValueError(f"Missing temporal prediction group: {path}")
        group = handle["temporal"]
        required = {"relation_ids", "sequences", "raw_logits", "targets"}
        missing = required - set(group.keys())
        if missing:
            raise ValueError(f"Missing prediction datasets in {path}: {sorted(missing)}")
        logits = np.asarray(group["raw_logits"], dtype=np.float32)
        targets = np.asarray(group["targets"], dtype=np.uint8)
        relation_ids = decode_array(group["relation_ids"])
        sequences = decode_array(group["sequences"])
        extras = {
            name: np.asarray(group[name])
            for name in (
                "tracklet_available",
                "predicted_source_rows",
                "predicted_target_rows",
                "predicted_joint_mask",
            )
            if name in group
        }
    if logits.ndim != 2 or logits.shape[1] != len(PREDICATES):
        raise ValueError(f"Unexpected temporal logit shape in {path}: {logits.shape}")
    if logits.shape != targets.shape or len(relation_ids) != len(logits):
        raise ValueError(f"Prediction rows are misaligned in {path}")
    if len(set(relation_ids)) != len(relation_ids):
        raise ValueError(f"Duplicate relation IDs in {path}")
    return PredictionBundle(relation_ids, sequences, logits, targets, extras)


def align_predictions(
    teacher: PredictionBundle, predicted: PredictionBundle
) -> tuple[PredictionBundle, PredictionBundle]:
    if set(teacher.relation_ids) != set(predicted.relation_ids):
        only_teacher = sorted(set(teacher.relation_ids) - set(predicted.relation_ids))[:5]
        only_predicted = sorted(set(predicted.relation_ids) - set(teacher.relation_ids))[:5]
        raise ValueError(
            "Teacher and predicted relation IDs differ: "
            f"teacher_only={only_teacher}, predicted_only={only_predicted}"
        )
    lookup = {relation_id: index for index, relation_id in enumerate(predicted.relation_ids)}
    order = np.asarray([lookup[value] for value in teacher.relation_ids], dtype=np.int64)
    aligned = PredictionBundle(
        relation_ids=list(teacher.relation_ids),
        sequences=[predicted.sequences[int(index)] for index in order],
        logits=predicted.logits[order],
        targets=predicted.targets[order],
        extras={name: value[order] for name, value in predicted.extras.items()},
    )
    if teacher.sequences != aligned.sequences:
        raise ValueError("Teacher and predicted sequence metadata differ")
    if not np.array_equal(teacher.targets, aligned.targets):
        raise ValueError("Teacher and predicted targets differ")
    return teacher, aligned


def read_feature_metadata(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any] | None] = [None] * expected_rows
    with path.resolve().open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            index = int(value["feature_row"])
            if not 0 <= index < expected_rows or rows[index] is not None:
                raise ValueError(f"Invalid or duplicate feature row {index} in {path}")
            if not value.get("instance_token_supervision_only"):
                raise ValueError(f"Feature row {index} lacks its evaluation-only identity")
            rows[index] = value
    if any(value is None for value in rows):
        raise ValueError(f"Feature metadata is incomplete: {path}")
    return [value for value in rows if value is not None]


def identity(metadata: list[dict[str, Any]], row: int) -> str | None:
    if row < 0:
        return None
    return str(metadata[row]["instance_token_supervision_only"])


def annotation_token(metadata: list[dict[str, Any]], row: int) -> str | None:
    if row < 0:
        return None
    return str(metadata[row]["annotation_token"])


def purity(values: Iterable[str | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(Counter(present).values()) / len(present)


def accepted_confidences(tracks: h5py.File) -> dict[tuple[int, int], float]:
    if "accepted_links" not in tracks:
        return {}
    links = tracks["accepted_links"]
    return {
        (int(source), int(target)): float(confidence)
        for source, target, confidence in zip(
            links["source_rows"], links["target_rows"], links["confidence"]
        )
    }


def consecutive_confidences(
    rows: np.ndarray, lookup: dict[tuple[int, int], float]
) -> list[float]:
    values: list[float] = []
    for source, target in zip(rows[:-1], rows[1:]):
        if source >= 0 and target >= 0 and (int(source), int(target)) in lookup:
            values.append(lookup[(int(source), int(target))])
    return values


def quality_bin(available: bool, correct_steps: int, identity_error: bool) -> str:
    if not available:
        return "missing"
    if identity_error:
        return "poor_identity"
    if correct_steps == 5:
        return "perfect_5"
    if correct_steps == 4:
        return "good_4"
    if correct_steps == 3:
        return "partial_3"
    return "poor_identity"


def unavailable_reason(
    source_track: int,
    target_track: int,
    joint_mask: np.ndarray,
) -> str:
    if source_track < 0 and target_track < 0:
        return "both_middle_rows_unassigned"
    if source_track < 0:
        return "source_middle_row_unassigned"
    if target_track < 0:
        return "target_middle_row_unassigned"
    if source_track == target_track:
        return "source_target_track_collision"
    if not bool(joint_mask[2]):
        return "middle_step_missing"
    if int(joint_mask.sum()) < 3:
        return "fewer_than_three_joint_steps"
    return "available"


def validate_relation_tokens(
    group: h5py.Group,
    row: int,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:
    if "source_annotation_tokens" not in group or "target_annotation_tokens" not in group:
        return
    source_tokens = [str(value) for value in group["source_annotation_tokens"].asstr()[row]]
    target_tokens = [str(value) for value in group["target_annotation_tokens"].asstr()[row]]
    for feature_row, stored in zip(source_rows, source_tokens):
        if feature_row >= 0 and annotation_token(metadata, int(feature_row)) != stored:
            raise ValueError("Source relation token does not match feature metadata")
    for feature_row, stored in zip(target_rows, target_tokens):
        if feature_row >= 0 and annotation_token(metadata, int(feature_row)) != stored:
            raise ValueError("Target relation token does not match feature metadata")


def analyze_sequence(
    metadata_path: Path,
    relation_path: Path,
    track_path: Path,
    prediction_lookup: dict[str, int],
    predicted: PredictionBundle,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with h5py.File(track_path.resolve(), "r", swmr=True) as tracks, h5py.File(
        relation_path.resolve(), "r", swmr=True
    ) as relations:
        if str(tracks.attrs.get("schema_version", "")) != "IBP-K360-predicted-tracklets-v1.1.0":
            raise ValueError(f"Unexpected track schema: {track_path}")
        if str(relations.attrs.get("schema_version", "")) != "IBP-K360-relation-index-v1.2.0":
            raise ValueError(f"Unexpected relation schema: {relation_path}")
        sequence = str(relations.attrs.get("sequence", ""))
        if not sequence or sequence != str(tracks.attrs.get("sequence", "")):
            raise ValueError(f"Sequence metadata differs for {relation_path} and {track_path}")
        row_track_ids = np.asarray(tracks["row_track_ids"], dtype=np.int64)
        row_frames = np.asarray(tracks["row_raw_frame_indices"], dtype=np.int64)
        metadata = read_feature_metadata(metadata_path, len(row_track_ids))
        if "row_annotation_tokens" in tracks:
            track_tokens = decode_array(tracks["row_annotation_tokens"])
            metadata_tokens = [str(value["annotation_token"]) for value in metadata]
            if track_tokens != metadata_tokens:
                raise ValueError(f"Track rows and feature metadata differ: {track_path}")

        members: dict[tuple[int, int], int] = {}
        for feature_row, (track_id, raw_frame) in enumerate(zip(row_track_ids, row_frames)):
            if track_id < 0:
                continue
            key = (int(track_id), int(raw_frame))
            if key in members:
                raise ValueError(f"Predicted track contains duplicate frame membership: {key}")
            members[key] = feature_row
        link_confidence = accepted_confidences(tracks)
        group = relations["temporal"]

        relation_ids = decode_array(group["relation_ids"])
        for relation_row, relation_id in enumerate(relation_ids):
            if relation_id not in prediction_lookup:
                raise ValueError(f"Relation {relation_id} is absent from prediction files")
            prediction_index = prediction_lookup[relation_id]
            if predicted.sequences[prediction_index] != sequence:
                raise ValueError(f"Prediction sequence differs for relation {relation_id}")
            relation_target = np.asarray(group["targets"][relation_row], dtype=np.uint8)
            if not np.array_equal(relation_target, predicted.targets[prediction_index]):
                raise ValueError(f"Relation targets differ for {relation_id}")

            frames = np.asarray(group["raw_frame_indices"][relation_row], dtype=np.int64)
            gt_source_rows = np.asarray(group["source_rows"][relation_row], dtype=np.int64)
            gt_target_rows = np.asarray(group["target_rows"][relation_row], dtype=np.int64)
            validate_relation_tokens(
                group, relation_row, gt_source_rows, gt_target_rows, metadata
            )
            source_middle = int(gt_source_rows[2])
            target_middle = int(gt_target_rows[2])
            source_track = int(row_track_ids[source_middle]) if source_middle >= 0 else -1
            target_track = int(row_track_ids[target_middle]) if target_middle >= 0 else -1
            predicted_source_rows = np.asarray(
                [members.get((source_track, int(frame)), -1) for frame in frames],
                dtype=np.int64,
            )
            predicted_target_rows = np.asarray(
                [members.get((target_track, int(frame)), -1) for frame in frames],
                dtype=np.int64,
            )
            predicted_joint = (predicted_source_rows >= 0) & (predicted_target_rows >= 0)
            reason = unavailable_reason(source_track, target_track, predicted_joint)
            available = reason == "available"

            source_expected = [identity(metadata, int(row)) for row in gt_source_rows]
            target_expected = [identity(metadata, int(row)) for row in gt_target_rows]
            source_actual = [identity(metadata, int(row)) for row in predicted_source_rows]
            target_actual = [identity(metadata, int(row)) for row in predicted_target_rows]
            source_correct = np.asarray(
                [expected is not None and actual == expected for expected, actual in zip(source_expected, source_actual)],
                dtype=np.bool_,
            )
            target_correct = np.asarray(
                [expected is not None and actual == expected for expected, actual in zip(target_expected, target_actual)],
                dtype=np.bool_,
            )
            joint_correct = source_correct & target_correct
            source_identity_switch = len({value for value in source_actual if value is not None}) > 1
            target_identity_switch = len({value for value in target_actual if value is not None}) > 1
            identity_error = bool(
                source_identity_switch
                or target_identity_switch
                or np.any((predicted_source_rows >= 0) & (gt_source_rows >= 0) & ~source_correct)
                or np.any((predicted_target_rows >= 0) & (gt_target_rows >= 0) & ~target_correct)
            )
            correct_steps = int(joint_correct.sum())
            teacher_joint = (gt_source_rows >= 0) & (gt_target_rows >= 0)
            confidences = consecutive_confidences(predicted_source_rows, link_confidence)
            confidences.extend(consecutive_confidences(predicted_target_rows, link_confidence))

            if "tracklet_available" in predicted.extras:
                stored_available = bool(predicted.extras["tracklet_available"][prediction_index])
                if stored_available != available:
                    raise ValueError(
                        f"Recomputed availability differs from saved output for {relation_id}"
                    )

            labels = [name for name, value in zip(PREDICATES, relation_target) if value]
            records.append(
                {
                    "prediction_index": prediction_index,
                    "relation_id": relation_id,
                    "sequence": sequence,
                    "group_id": str(group["group_ids"].asstr()[relation_row]),
                    "start_raw_frame": int(frames[0]),
                    "middle_raw_frame": int(frames[2]),
                    "end_raw_frame": int(frames[-1]),
                    "labels": "|".join(labels) if labels else "no_relation",
                    "teacher_joint_steps": int(teacher_joint.sum()),
                    "predicted_source_steps": int((predicted_source_rows >= 0).sum()),
                    "predicted_target_steps": int((predicted_target_rows >= 0).sum()),
                    "predicted_joint_steps": int(predicted_joint.sum()),
                    "correct_source_identity_steps": int(source_correct.sum()),
                    "correct_target_identity_steps": int(target_correct.sum()),
                    "correct_joint_identity_steps": correct_steps,
                    "relative_joint_identity_recall": correct_steps / max(int(teacher_joint.sum()), 1),
                    "source_identity_purity": purity(source_actual),
                    "target_identity_purity": purity(target_actual),
                    "source_identity_switch": source_identity_switch,
                    "target_identity_switch": target_identity_switch,
                    "identity_error": identity_error,
                    "available": available,
                    "unavailable_reason": reason,
                    "quality_bin": quality_bin(available, correct_steps, identity_error),
                    "mean_accepted_link_confidence": float(np.mean(confidences)) if confidences else None,
                    "minimum_accepted_link_confidence": float(np.min(confidences)) if confidences else None,
                }
            )
    return records


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order].astype(np.float64)
    precision = np.cumsum(sorted_target) / np.arange(1, len(target) + 1)
    return float((precision * sorted_target).sum() / positives)


def multilabel_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if len(logits) == 0:
        return {
            "examples": 0,
            "macro_f1": None,
            "mean_average_precision": None,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "exact_match_accuracy": None,
            "per_predicate": {},
        }
    probabilities = sigmoid(logits)
    predictions = probabilities >= 0.5
    target_bool = targets.astype(np.bool_)
    per_predicate: dict[str, dict[str, Any]] = {}
    supported_f1: list[float] = []
    supported_ap: list[float] = []
    for index, name in enumerate(PREDICATES):
        actual = target_bool[:, index]
        predicted = predictions[:, index]
        true_positive = int(np.logical_and(actual, predicted).sum())
        false_positive = int(np.logical_and(~actual, predicted).sum())
        false_negative = int(np.logical_and(actual, ~predicted).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        ap = average_precision(targets[:, index], probabilities[:, index])
        support = int(actual.sum())
        if support:
            supported_f1.append(f1)
        if ap is not None:
            supported_ap.append(ap)
        per_predicate[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": ap,
        }
    flat_actual = target_bool.reshape(-1)
    flat_predicted = predictions.reshape(-1)
    true_positive = int(np.logical_and(flat_actual, flat_predicted).sum())
    false_positive = int(np.logical_and(~flat_actual, flat_predicted).sum())
    false_negative = int(np.logical_and(flat_actual, ~flat_predicted).sum())
    micro_precision = true_positive / max(true_positive + false_positive, 1)
    micro_recall = true_positive / max(true_positive + false_negative, 1)
    micro_f1 = 2.0 * micro_precision * micro_recall / max(
        micro_precision + micro_recall, 1e-12
    )
    return {
        "examples": int(len(logits)),
        "macro_f1": float(np.mean(supported_f1)) if supported_f1 else None,
        "mean_average_precision": float(np.mean(supported_ap)) if supported_ap else None,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_match_accuracy": float(np.all(target_bool == predictions, axis=1).mean()),
        "per_predicate": per_predicate,
    }


def difference(predicted: float | None, teacher: float | None) -> float | None:
    if predicted is None or teacher is None:
        return None
    return float(predicted - teacher)


def summarize_indices(
    indices: np.ndarray,
    teacher: PredictionBundle,
    predicted: PredictionBundle,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    teacher_metrics = multilabel_metrics(teacher.logits[indices], teacher.targets[indices])
    predicted_metrics = multilabel_metrics(predicted.logits[indices], predicted.targets[indices])
    selected = [records[int(index)] for index in indices]
    available = [bool(record["available"]) for record in selected]
    identity_errors = [bool(record["identity_error"]) for record in selected]
    joint_steps = [int(record["correct_joint_identity_steps"]) for record in selected]
    return {
        "examples": int(len(indices)),
        "positive_labels": int(teacher.targets[indices].sum()) if len(indices) else 0,
        "tracklet_availability": float(np.mean(available)) if available else None,
        "identity_error_rate": float(np.mean(identity_errors)) if identity_errors else None,
        "mean_correct_joint_identity_steps": float(np.mean(joint_steps)) if joint_steps else None,
        "teacher_forced": teacher_metrics,
        "predicted_tracklet": predicted_metrics,
        "predicted_minus_teacher": {
            "macro_f1": difference(
                predicted_metrics["macro_f1"], teacher_metrics["macro_f1"]
            ),
            "mean_average_precision": difference(
                predicted_metrics["mean_average_precision"],
                teacher_metrics["mean_average_precision"],
            ),
            "micro_recall": difference(
                predicted_metrics["micro_recall"], teacher_metrics["micro_recall"]
            ),
        },
    }


def write_records(
    path: Path,
    records: list[dict[str, Any]],
    teacher: PredictionBundle,
    predicted: PredictionBundle,
) -> None:
    fieldnames = [key for key in records[0] if key != "prediction_index"]
    for prefix in ("teacher_logit", "predicted_logit"):
        fieldnames.extend(f"{prefix}_{name}" for name in PREDICATES)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            index = int(record["prediction_index"])
            output = {key: value for key, value in record.items() if key != "prediction_index"}
            for predicate_index, name in enumerate(PREDICATES):
                output[f"teacher_logit_{name}"] = float(teacher.logits[index, predicate_index])
                output[f"predicted_logit_{name}"] = float(predicted.logits[index, predicate_index])
            writer.writerow(output)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    sections = [("ALL", summary["overall"], summary["by_quality"])]
    sections.extend(
        (sequence, value["overall"], value["by_quality"])
        for sequence, value in summary["by_sequence"].items()
    )
    for sequence, overall, quality in sections:
        for name, value in [("overall", overall), *quality.items()]:
            rows.append(
                {
                    "sequence": sequence,
                    "quality_bin": name,
                    "examples": value["examples"],
                    "positive_labels": value["positive_labels"],
                    "tracklet_availability": value["tracklet_availability"],
                    "identity_error_rate": value["identity_error_rate"],
                    "teacher_macro_f1": value["teacher_forced"]["macro_f1"],
                    "predicted_macro_f1": value["predicted_tracklet"]["macro_f1"],
                    "teacher_map": value["teacher_forced"]["mean_average_precision"],
                    "predicted_map": value["predicted_tracklet"]["mean_average_precision"],
                    "predicted_minus_teacher_macro_f1": value["predicted_minus_teacher"]["macro_f1"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, by_quality: dict[str, Any]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = [name for name in QUALITY_BIN_ORDER if by_quality[name]["examples"]]
    labels = [name.replace("_", "\n") for name in names]
    teacher_f1 = [
        100.0 * float(by_quality[name]["teacher_forced"]["macro_f1"] or 0.0)
        for name in names
    ]
    predicted_f1 = [
        100.0 * float(by_quality[name]["predicted_tracklet"]["macro_f1"] or 0.0)
        for name in names
    ]
    counts = [int(by_quality[name]["examples"]) for name in names]
    x = np.arange(len(names))
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].bar(x - 0.2, teacher_f1, 0.4, label="Teacher-forced")
    axes[0].bar(x + 0.2, predicted_f1, 0.4, label="Predicted tracklets")
    axes[0].set_ylabel("Temporal macro-F1 (%)")
    axes[0].set_title("Temporal performance by predicted-tracklet quality")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, counts)
    axes[1].set_ylabel("Relation examples")
    axes[1].set_xlabel("Tracklet-quality bin")
    axes[1].set_title("Support in each quality bin")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    metrics_path = output_root / "tracklet_quality_metrics.json"
    if metrics_path.exists() and not args.force:
        raise FileExistsError(f"Analysis output already exists: {metrics_path}")
    counts = {
        len(args.feature_metadata),
        len(args.relation_files),
        len(args.track_files),
    }
    if len(counts) != 1:
        raise ValueError("Feature metadata, relation files, and track files must align")
    teacher, predicted = align_predictions(
        load_predictions(args.teacher_predictions),
        load_predictions(args.predicted_predictions),
    )
    lookup = {relation_id: index for index, relation_id in enumerate(teacher.relation_ids)}
    records: list[dict[str, Any]] = []
    for metadata, relations, tracks in zip(
        args.feature_metadata, args.relation_files, args.track_files
    ):
        records.extend(
            analyze_sequence(metadata, relations, tracks, lookup, predicted)
        )
    if not records:
        raise ValueError("Tracklet analysis produced no relation records")
    if len(records) != len(teacher.relation_ids):
        raise ValueError(
            f"Analyzed {len(records)} records for {len(teacher.relation_ids)} predictions"
        )
    records.sort(key=lambda value: int(value["prediction_index"]))
    if [record["relation_id"] for record in records] != teacher.relation_ids:
        raise ValueError("Relation-file rows do not reproduce prediction order")

    all_indices = np.arange(len(records), dtype=np.int64)
    by_quality = {
        name: summarize_indices(
            np.asarray(
                [index for index, record in enumerate(records) if record["quality_bin"] == name],
                dtype=np.int64,
            ),
            teacher,
            predicted,
            records,
        )
        for name in QUALITY_BIN_ORDER
    }
    by_sequence: dict[str, Any] = {}
    for sequence in sorted(set(teacher.sequences)):
        indices = np.asarray(
            [index for index, record in enumerate(records) if record["sequence"] == sequence],
            dtype=np.int64,
        )
        by_sequence[sequence] = {
            "overall": summarize_indices(indices, teacher, predicted, records),
            "by_quality": {
                name: summarize_indices(
                    np.asarray(
                        [index for index in indices if records[int(index)]["quality_bin"] == name],
                        dtype=np.int64,
                    ),
                    teacher,
                    predicted,
                    records,
                )
                for name in QUALITY_BIN_ORDER
            },
        }

    failure_reasons = Counter(
        str(record["unavailable_reason"])
        for record in records
        if not bool(record["available"])
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split_name,
        "sequences": sorted(set(teacher.sequences)),
        "quality_bin_definitions": {
            "perfect_5": "Available, correct subject and object identities at all five steps, no identity error.",
            "good_4": "Available, correct subject and object identities at four steps, no identity error.",
            "partial_3": "Available, correct subject and object identities at three steps, no identity error.",
            "poor_identity": "Marked available but contains an identity switch, wrong identity, or fewer than three correct joint identity steps.",
            "missing": "Unavailable to the temporal model under the current predicted-tracklet rule.",
        },
        "overall": summarize_indices(all_indices, teacher, predicted, records),
        "available_only": summarize_indices(
            np.asarray(
                [index for index, record in enumerate(records) if record["available"]],
                dtype=np.int64,
            ),
            teacher,
            predicted,
            records,
        ),
        "by_quality": by_quality,
        "by_sequence": by_sequence,
        "quality_counts": Counter(str(record["quality_bin"]) for record in records),
        "unavailable_reason_counts": dict(sorted(failure_reasons.items())),
        "evaluation_only_identity_note": (
            "Persistent KITTI-360 identities are used only for this post-hoc quality audit, "
            "never as model inputs or association features."
        ),
        "test_use_policy": (
            "Descriptive held-out analysis only; do not select thresholds or hyperparameters "
            "from this output."
            if args.split_name == "held_out_test"
            else "Validation analysis may be used for method development before freezing changes."
        ),
        "inputs": {
            "teacher_predictions": str(args.teacher_predictions.resolve()),
            "teacher_predictions_sha256": sha256_file(args.teacher_predictions.resolve()),
            "predicted_predictions": str(args.predicted_predictions.resolve()),
            "predicted_predictions_sha256": sha256_file(args.predicted_predictions.resolve()),
            "relation_files": [str(path.resolve()) for path in args.relation_files],
            "track_files": [str(path.resolve()) for path in args.track_files],
        },
    }
    summary["quality_counts"] = dict(summary["quality_counts"])
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "tracklet_quality_records.csv.gz"
    summary_csv_path = output_root / "tracklet_quality_summary.csv"
    plot_path = output_root / "tracklet_quality_curve.png"
    write_records(records_path, records, teacher, predicted)
    write_summary_csv(summary_csv_path, summary)
    summary["plot_created"] = write_plot(plot_path, by_quality)
    summary["records_file"] = str(records_path)
    summary["summary_csv"] = str(summary_csv_path)
    if summary["plot_created"]:
        summary["plot"] = str(plot_path)
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(metrics_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify temporal predicate degradation by predicted-tracklet quality."
    )
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--predicted-predictions", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--relation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--track-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-name", choices=("validation", "held_out_test"), required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), indent=2))


if __name__ == "__main__":
    main()
