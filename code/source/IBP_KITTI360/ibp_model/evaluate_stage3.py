from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import multilabel_metrics
from .model import IBPK360Model
from .predicted_relation_data import PredictedTemporalRelationDataset
from .predicted_tracklets import feature_batch, read_feature_metadata, sha256_file
from .protocol import (
    ModelConfig,
    OBJECT_CLASSES,
    SPATIAL_PREDICATES,
    TEMPORAL_PREDICATES,
    validate_protocol,
)
from .relation_data import SpatialRelationDataset
from .train_stage3_predicted import association_summary, spatial_step, temporal_step


SCHEMA_VERSION = "IBP-K360-final-predicted-graph-evaluation-v1.0.0"


def relation_metadata(
    relation_paths: list[Path], group_name: str
) -> dict[str, Any]:
    relation_ids: list[str] = []
    group_ids: list[str] = []
    sequences: list[str] = []
    source_tokens: list[np.ndarray] = []
    target_tokens: list[np.ndarray] = []
    raw_frames: list[np.ndarray] = []
    for path in relation_paths:
        with h5py.File(path.resolve(), "r", swmr=True) as relations:
            if str(relations.attrs.get("schema_version", "")) != "IBP-K360-relation-index-v1.2.0":
                raise ValueError(f"Stage-3 evaluation requires relation-index v1.2: {path}")
            group = relations[group_name]
            count = len(group["targets"])
            relation_ids.extend(group["relation_ids"].asstr()[:].tolist())
            group_ids.extend(group["group_ids"].asstr()[:].tolist())
            sequences.extend([str(relations.attrs.get("sequence", ""))] * count)
            source_tokens.append(group["source_annotation_tokens"].asstr()[:])
            target_tokens.append(group["target_annotation_tokens"].asstr()[:])
            raw_frames.append(np.asarray(group["raw_frame_indices"], dtype=np.int64))
    return {
        "relation_ids": relation_ids,
        "group_ids": group_ids,
        "sequences": sequences,
        "source_annotation_tokens": np.concatenate(source_tokens),
        "target_annotation_tokens": np.concatenate(target_tokens),
        "raw_frame_indices": np.concatenate(raw_frames),
    }


def object_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    prediction = logits.argmax(dim=1)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(OBJECT_CLASSES):
        actual = labels.eq(index)
        predicted = prediction.eq(index)
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        support = int(actual.sum())
        if support:
            f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "objects": int(len(labels)),
        "accuracy": float(prediction.eq(labels).float().mean()),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def predicate_recall_at_k(
    logits: np.ndarray,
    labels: np.ndarray,
    group_ids: list[str],
    predicate_names: tuple[str, ...],
    ks: tuple[int, ...] = (20, 50, 100),
) -> dict[str, Any]:
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)
    total_positive = int(labels.sum())
    support = labels.sum(axis=0, dtype=np.int64)
    result: dict[str, Any] = {
        "definition": "Predicate ranking over fixed ordered object-pair candidates with ground-truth objects.",
        "positive_triplets": total_positive,
    }
    for k in ks:
        recovered = 0
        recovered_by_class = np.zeros(len(predicate_names), dtype=np.int64)
        for indices in grouped.values():
            local_scores = scores[indices].reshape(-1)
            keep = min(k, len(local_scores))
            if not keep:
                continue
            selected = np.argpartition(-local_scores, keep - 1)[:keep]
            pair_indices = selected // len(predicate_names)
            predicate_indices = selected % len(predicate_names)
            for pair_index, predicate_index in zip(pair_indices, predicate_indices):
                if labels[indices[int(pair_index)], int(predicate_index)]:
                    recovered += 1
                    recovered_by_class[int(predicate_index)] += 1
        recalls = [
            recovered_by_class[index] / int(support[index])
            for index in range(len(predicate_names))
            if support[index] > 0
        ]
        result[f"R@{k}"] = recovered / max(total_positive, 1)
        result[f"mR@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        result[f"per_predicate_R@{k}"] = {
            name: recovered_by_class[index] / max(int(support[index]), 1)
            for index, name in enumerate(predicate_names)
        }
    return result


def write_prediction_group(
    group: h5py.Group,
    logits: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    group_ids: list[str],
    sequences: list[str],
    predicate_names: tuple[str, ...],
    **extra: np.ndarray,
) -> None:
    group.attrs["predicate_names"] = json.dumps(list(predicate_names))
    group.create_dataset("raw_logits", data=logits.astype(np.float32), compression="gzip")
    group.create_dataset("targets", data=labels.astype(np.uint8), compression="gzip")
    group.create_dataset("relation_ids", data=np.asarray(ids, dtype="S32"), compression="gzip")
    group.create_dataset("group_ids", data=np.asarray(group_ids, dtype="S96"), compression="gzip")
    group.create_dataset("sequences", data=np.asarray(sequences, dtype="S32"), compression="gzip")
    for name, value in extra.items():
        group.create_dataset(name, data=value, compression="gzip")


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    final_metrics_path = output_root / "final_test_metrics.json"
    if final_metrics_path.exists() and not args.force:
        raise FileExistsError(
            f"Final test metrics already exist: {final_metrics_path}. "
            "Refusing to repeat the held-out test."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not bool(checkpoint.get("final_predicted_tracklet_checkpoint", False)):
        raise ValueError("Final evaluation requires a predicted-tracklet Stage-3 checkpoint.")
    association_checkpoint_sha256 = str(
        checkpoint.get("association_source_checkpoint_sha256", "")
    )
    if not association_checkpoint_sha256:
        raise ValueError("Stage-3 checkpoint does not record its association checkpoint hash.")
    for path in args.test_tracks:
        with h5py.File(path.resolve(), "r", swmr=True) as tracks:
            if str(tracks.attrs.get("source_checkpoint_sha256", "")) != (
                association_checkpoint_sha256
            ):
                raise ValueError(
                    f"Test tracks use a different association checkpoint: {path}"
                )
    model_config = ModelConfig(**checkpoint["model_config"])
    validate_protocol(model_config)
    model = IBPK360Model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    object_logits: list[torch.Tensor] = []
    object_labels: list[torch.Tensor] = []
    object_ids: list[str] = []
    object_sequences: list[str] = []
    for feature_path, metadata_path in zip(args.test_features, args.test_metadata):
        with h5py.File(feature_path.resolve(), "r", swmr=True) as features:
            metadata = read_feature_metadata(metadata_path.resolve(), len(features["category_id"]))
            rows = np.flatnonzero(
                np.asarray(features["modality_mask"], dtype=np.bool_).all(axis=1)
            ).astype(np.int64)
            for start in tqdm(
                range(0, len(rows), args.batch_size),
                desc=f"Objects {feature_path.stem}",
            ):
                selected = rows[start : start + args.batch_size]
                encoded = model.encode_objects(**feature_batch(features, selected, device))
                object_logits.append(model.predict_nodes(encoded["fused_object"]).cpu())
                object_labels.append(
                    torch.from_numpy(np.asarray(features["category_id"][selected], dtype=np.int64))
                )
                object_ids.extend(str(metadata[int(row)]["annotation_token"]) for row in selected)
                object_sequences.extend(str(metadata[int(row)]["sequence"]) for row in selected)

    spatial_dataset = SpatialRelationDataset(args.test_features, args.test_relations)
    spatial_loader = DataLoader(spatial_dataset, batch_size=args.batch_size, shuffle=False)
    spatial_logits: list[torch.Tensor] = []
    spatial_labels: list[torch.Tensor] = []
    unit_spatial_weight = torch.ones(len(SPATIAL_PREDICATES), device=device)
    unit_object_weight = torch.ones(len(OBJECT_CLASSES), device=device)
    for batch in tqdm(spatial_loader, desc="Spatial test"):
        _, logits, labels = spatial_step(
            model, batch, device, unit_spatial_weight, unit_object_weight
        )
        spatial_logits.append(logits.cpu())
        spatial_labels.append(labels.cpu())
    spatial_metadata = relation_metadata(args.test_relations, "spatial")
    spatial_groups = [
        f"{sequence}:{group}"
        for sequence, group in zip(
            spatial_metadata["sequences"], spatial_metadata["group_ids"]
        )
    ]

    temporal_dataset = PredictedTemporalRelationDataset(
        args.test_features, args.test_relations, args.test_tracks
    )
    temporal_loader = DataLoader(temporal_dataset, batch_size=args.batch_size, shuffle=False)
    temporal_logits: list[torch.Tensor] = []
    temporal_labels: list[torch.Tensor] = []
    temporal_available: list[torch.Tensor] = []
    predicted_source_rows: list[torch.Tensor] = []
    predicted_target_rows: list[torch.Tensor] = []
    unit_temporal_weight = torch.ones(len(TEMPORAL_PREDICATES), device=device)
    for batch in tqdm(temporal_loader, desc="Temporal predicted-track test"):
        _, logits, labels = temporal_step(model, batch, device, unit_temporal_weight)
        available = batch["available"].bool()
        logits = logits.cpu()
        logits[~available] = -20.0
        temporal_logits.append(logits)
        temporal_labels.append(labels.cpu())
        temporal_available.append(available)
        predicted_source_rows.append(batch["predicted_source_rows"].long())
        predicted_target_rows.append(batch["predicted_target_rows"].long())
    temporal_metadata = relation_metadata(args.test_relations, "temporal")
    temporal_groups = [
        f"{sequence}:{group}"
        for sequence, group in zip(
            temporal_metadata["sequences"], temporal_metadata["group_ids"]
        )
    ]

    object_logits_tensor = torch.cat(object_logits)
    object_labels_tensor = torch.cat(object_labels)
    spatial_logits_tensor = torch.cat(spatial_logits)
    spatial_labels_tensor = torch.cat(spatial_labels)
    temporal_logits_tensor = torch.cat(temporal_logits)
    temporal_labels_tensor = torch.cat(temporal_labels)
    temporal_available_tensor = torch.cat(temporal_available)
    association = association_summary(
        [path.with_suffix(".report.json") for path in args.test_tracks]
    )
    spatial_metrics = multilabel_metrics(
        spatial_logits_tensor, spatial_labels_tensor, SPATIAL_PREDICATES
    )
    temporal_metrics = multilabel_metrics(
        temporal_logits_tensor, temporal_labels_tensor, TEMPORAL_PREDICATES
    )
    supported_f1 = [
        float(value["f1"])
        for metrics in (spatial_metrics, temporal_metrics)
        for value in metrics["per_predicate"].values()
        if int(value["support"]) > 0
    ]
    supported_ap = [
        float(value["average_precision"])
        for metrics in (spatial_metrics, temporal_metrics)
        for value in metrics["per_predicate"].values()
        if value["average_precision"] is not None
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "split": "held_out_test",
        "enabled_modalities": list(model_config.enabled_modalities),
        "use_parts": model_config.use_parts,
        "test_sequences": sorted(set(object_sequences)),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "object": object_metrics(object_logits_tensor, object_labels_tensor),
        "spatial": spatial_metrics,
        "temporal_end_to_end": temporal_metrics,
        "predicted_tracklet_coverage": float(temporal_available_tensor.float().mean()),
        "association": association,
        "spatial_triplet_ranking": predicate_recall_at_k(
            spatial_logits_tensor.numpy(),
            spatial_labels_tensor.numpy().astype(np.uint8),
            spatial_groups,
            SPATIAL_PREDICATES,
        ),
        "temporal_triplet_ranking": predicate_recall_at_k(
            temporal_logits_tensor.numpy(),
            temporal_labels_tensor.numpy().astype(np.uint8),
            temporal_groups,
            TEMPORAL_PREDICATES,
        ),
        "combined_eight_predicate_macro_f1": float(np.mean(supported_f1)),
        "combined_eight_predicate_map": float(np.mean(supported_ap)),
        "threshold": 0.5,
        "teacher_forced_tracklets": False,
        "publishable_final_result": True,
    }

    prediction_path = output_root / "raw_predictions.h5"
    temporary = prediction_path.with_suffix(".h5.tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["checkpoint_sha256"] = metrics["checkpoint_sha256"]
        objects = output.create_group("objects")
        objects.attrs["class_names"] = json.dumps(list(OBJECT_CLASSES))
        objects.create_dataset(
            "raw_logits", data=object_logits_tensor.numpy().astype(np.float32), compression="gzip"
        )
        objects.create_dataset(
            "targets", data=object_labels_tensor.numpy().astype(np.int16), compression="gzip"
        )
        objects.create_dataset(
            "annotation_tokens", data=np.asarray(object_ids, dtype="S32"), compression="gzip"
        )
        objects.create_dataset(
            "sequences", data=np.asarray(object_sequences, dtype="S32"), compression="gzip"
        )
        write_prediction_group(
            output.create_group("spatial"),
            spatial_logits_tensor.numpy(),
            spatial_labels_tensor.numpy(),
            spatial_metadata["relation_ids"],
            spatial_groups,
            spatial_metadata["sequences"],
            SPATIAL_PREDICATES,
            source_annotation_tokens=spatial_metadata["source_annotation_tokens"],
            target_annotation_tokens=spatial_metadata["target_annotation_tokens"],
            raw_frame_indices=spatial_metadata["raw_frame_indices"],
        )
        write_prediction_group(
            output.create_group("temporal"),
            temporal_logits_tensor.numpy(),
            temporal_labels_tensor.numpy(),
            temporal_metadata["relation_ids"],
            temporal_groups,
            temporal_metadata["sequences"],
            TEMPORAL_PREDICATES,
            source_annotation_tokens=temporal_metadata["source_annotation_tokens"],
            target_annotation_tokens=temporal_metadata["target_annotation_tokens"],
            raw_frame_indices=temporal_metadata["raw_frame_indices"],
            tracklet_available=temporal_available_tensor.numpy().astype(np.uint8),
            predicted_source_rows=torch.cat(predicted_source_rows).numpy().astype(np.int64),
            predicted_target_rows=torch.cat(predicted_target_rows).numpy().astype(np.int64),
        )
    temporary.replace(prediction_path)
    metrics["raw_predictions"] = str(prediction_path)
    final_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    spatial_dataset.close()
    temporal_dataset.close()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-time held-out KITTI-360 predicted-graph evaluation."
    )
    parser.add_argument("--test-features", type=Path, nargs="+", required=True)
    parser.add_argument("--test-metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--test-relations", type=Path, nargs="+", required=True)
    parser.add_argument("--test-tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sizes = {
        len(args.test_features),
        len(args.test_metadata),
        len(args.test_relations),
        len(args.test_tracks),
    }
    if len(sizes) != 1:
        parser.error("Test feature, metadata, relation, and track shard counts must match")
    return args


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
