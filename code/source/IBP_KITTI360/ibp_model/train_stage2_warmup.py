from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from .losses import (
    association_cross_entropy,
    multilabel_focal_loss,
    representation_losses,
    temporal_consistency,
)
from .metrics import multilabel_metrics
from .model import IBPK360Model
from .protocol import (
    ModelConfig,
    OBJECT_CLASSES,
    SPATIAL_PREDICATES,
    TEMPORAL_PREDICATES,
    validate_protocol,
)
from .relation_data import AssociationDataset, SpatialRelationDataset, TemporalRelationDataset
from .train_stage1 import object_class_statistics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_object(value: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value[key].to(device, non_blocking=True)
        for key in ("rgb_tokens", "lidar_tokens", "lidar_anchor", "text_embedding", "modality_mask")
    }


def join_objects(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {key: torch.cat([first[key], second[key]], dim=0) for key in first}


def split_encoded(outputs: dict[str, torch.Tensor], first_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {key: value[:first_size] for key, value in outputs.items()},
        {key: value[first_size:] for key, value in outputs.items()},
    )


def next_or_none(iterator: Any, loader: DataLoader | None) -> tuple[Any, Any]:
    if loader is None:
        return None, iterator
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def multilabel_macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = torch.sigmoid(logits).ge(0.5)
    actual = labels.bool()
    scores = []
    for index in range(labels.size(1)):
        tp = (predicted[:, index] & actual[:, index]).sum().item()
        fp = (predicted[:, index] & ~actual[:, index]).sum().item()
        fn = (~predicted[:, index] & actual[:, index]).sum().item()
        denominator = 2 * tp + fp + fn
        if actual[:, index].any():
            scores.append(2 * tp / denominator if denominator else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def predicate_pos_weight(
    relation_paths: list[Path], group_name: str, classes: int, maximum: float
) -> tuple[torch.Tensor, list[int], int]:
    """Estimate capped positive weights from training relation shards only."""
    positives = np.zeros(classes, dtype=np.int64)
    total = 0
    for path in relation_paths:
        with h5py.File(path.resolve(), "r") as relation_file:
            targets = np.asarray(relation_file[group_name]["targets"], dtype=np.uint8)
        if targets.ndim != 2 or targets.shape[1] != classes:
            raise ValueError(f"Unexpected {group_name} target shape in {path}: {targets.shape}")
        positives += targets.sum(axis=0, dtype=np.int64)
        total += len(targets)
    if total == 0:
        return torch.ones(classes), positives.tolist(), total
    negatives = total - positives
    weights = np.ones(classes, dtype=np.float32)
    supported = positives > 0
    weights[supported] = negatives[supported] / positives[supported]
    weights = np.clip(weights, 1.0, maximum)
    return torch.from_numpy(weights), positives.tolist(), total


def encode_tracklets(
    model: IBPK360Model,
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch, steps = source["valid_mask"].shape
    flattened_source = {
        key: source[key].reshape(batch * steps, *source[key].shape[2:])
        for key in ("rgb_tokens", "lidar_tokens", "lidar_anchor", "text_embedding", "modality_mask")
    }
    flattened_target = {
        key: target[key].reshape(batch * steps, *target[key].shape[2:])
        for key in flattened_source
    }
    joined = join_objects(move_object(flattened_source, device), move_object(flattened_target, device))
    encoded = model.encode_objects(**joined)
    source_encoded, target_encoded = split_encoded(encoded, batch * steps)
    source_encoded = {
        key: value.reshape(batch, steps, *value.shape[1:]) for key, value in source_encoded.items()
    }
    target_encoded = {
        key: value.reshape(batch, steps, *value.shape[1:]) for key, value in target_encoded.items()
    }
    source_mask = source["valid_mask"].to(device)
    target_mask = target["valid_mask"].to(device)
    return source_encoded, target_encoded, source_mask, target_mask


def spatial_step(
    model: IBPK360Model,
    batch: dict[str, Any],
    device: torch.device,
    spatial_pos_weight: torch.Tensor | None = None,
    object_class_weight: torch.Tensor | None = None,
):
    source = move_object(batch["source"], device)
    target = move_object(batch["target"], device)
    joined = join_objects(source, target)
    outputs = model.encode_objects(**joined)
    count = source["rgb_tokens"].size(0)
    first, second = split_encoded(outputs, count)
    logits = model.predict_spatial(
        first["fused_object"],
        second["fused_object"],
        first["fused_parts"],
        second["fused_parts"],
        batch["geometry"].to(device),
    )
    labels = batch["labels"].to(device)
    node_logits = model.predict_nodes(torch.cat([first["fused_object"], second["fused_object"]]))
    node_labels = torch.cat([batch["source"]["category_id"], batch["target"]["category_id"]]).to(device)
    rep = representation_losses(outputs)
    return {
        "edge": multilabel_focal_loss(logits, labels, pos_weight=spatial_pos_weight),
        "node": F.cross_entropy(node_logits, node_labels, weight=object_class_weight),
        "diversity": rep["diversity"],
        "part": rep["part_alignment"],
        "object": rep["object_alignment"],
    }, logits.detach(), labels.detach()


def temporal_step(
    model: IBPK360Model,
    batch: dict[str, Any],
    device: torch.device,
    temporal_pos_weight: torch.Tensor | None = None,
):
    source, target, source_mask, target_mask = encode_tracklets(
        model, batch["source"], batch["target"], device
    )
    middle = source["fused_object"].size(1) // 2
    logits = model.predict_temporal(
        source["fused_object"],
        target["fused_object"],
        source_mask,
        target_mask,
        source["fused_parts"][:, middle],
        target["fused_parts"][:, middle],
        batch["geometry"].to(device),
    )
    labels = batch["labels"].to(device)
    source_context = model.temporal(source["fused_object"], source_mask)
    target_context = model.temporal(target["fused_object"], target_mask)
    consistency = 0.5 * (
        temporal_consistency(source_context, source_mask)
        + temporal_consistency(target_context, target_mask)
    )
    return {
        "edge": multilabel_focal_loss(logits, labels, pos_weight=temporal_pos_weight),
        "consistency": consistency,
    }, logits.detach(), labels.detach()


def association_step(model: IBPK360Model, batch: dict[str, Any], device: torch.device):
    source = model.encode_objects(**move_object(batch["source"], device))["fused_object"]
    target = model.encode_objects(**move_object(batch["target"], device))["fused_object"]
    outputs = model.association(
        source,
        target,
        batch["geometry"].to(device),
        batch["candidate_mask"].to(device),
    )
    loss = association_cross_entropy(
        outputs["augmented_logits"],
        batch["source_targets"].to(device),
        batch["target_targets"].to(device),
    )
    source_prediction = outputs["augmented_logits"][:-1].argmax(dim=1)
    source_actual = batch["source_targets"].to(device)
    target_prediction = outputs["augmented_logits"][:, :-1].argmax(dim=0)
    target_actual = batch["target_targets"].to(device)
    correct = source_prediction.eq(source_actual).sum() + target_prediction.eq(target_actual).sum()
    total = source_actual.numel() + target_actual.numel()
    return loss, int(correct), int(total)


def make_loaders(
    features: list[Path], relations: list[Path], batch_size: int, shuffle: bool
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None, list[Any]]:
    spatial = SpatialRelationDataset(features, relations)
    temporal = TemporalRelationDataset(features, relations)
    association = AssociationDataset(features, relations)
    spatial_loader = DataLoader(spatial, batch_size=batch_size, shuffle=shuffle) if len(spatial) else None
    temporal_loader = DataLoader(temporal, batch_size=batch_size, shuffle=shuffle) if len(temporal) else None
    association_loader = (
        DataLoader(association, batch_size=1, shuffle=shuffle, collate_fn=lambda values: values[0])
        if len(association)
        else None
    )
    return spatial_loader, temporal_loader, association_loader, [spatial, temporal, association]


@torch.inference_mode()
def evaluate(
    model: IBPK360Model,
    loaders: tuple[DataLoader | None, DataLoader | None, DataLoader | None],
    device: torch.device,
    max_batches: int | None,
    spatial_pos_weight: torch.Tensor,
    temporal_pos_weight: torch.Tensor,
    object_class_weight: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    spatial_logits, spatial_labels = [], []
    temporal_logits, temporal_labels = [], []
    association_correct = association_total = 0
    if loaders[0] is not None:
        for index, batch in enumerate(loaders[0]):
            if max_batches is not None and index >= max_batches:
                break
            _, logits, labels = spatial_step(
                model, batch, device, spatial_pos_weight, object_class_weight
            )
            spatial_logits.append(logits.cpu())
            spatial_labels.append(labels.cpu())
    if loaders[1] is not None:
        for index, batch in enumerate(loaders[1]):
            if max_batches is not None and index >= max_batches:
                break
            _, logits, labels = temporal_step(model, batch, device, temporal_pos_weight)
            temporal_logits.append(logits.cpu())
            temporal_labels.append(labels.cpu())
    if loaders[2] is not None:
        for index, batch in enumerate(loaders[2]):
            if max_batches is not None and index >= max_batches:
                break
            _, correct, total = association_step(model, batch, device)
            association_correct += correct
            association_total += total
    spatial_metrics = (
        multilabel_metrics(
            torch.cat(spatial_logits), torch.cat(spatial_labels), SPATIAL_PREDICATES
        )
        if spatial_logits
        else None
    )
    temporal_metrics = (
        multilabel_metrics(
            torch.cat(temporal_logits), torch.cat(temporal_labels), TEMPORAL_PREDICATES
        )
        if temporal_logits
        else None
    )
    association_accuracy = association_correct / association_total if association_total else 0.0
    available = []
    if spatial_metrics is not None:
        available.append(float(spatial_metrics["mean_average_precision"] or 0.0))
    if temporal_metrics is not None:
        available.append(float(temporal_metrics["mean_average_precision"] or 0.0))
    if association_total:
        available.append(association_accuracy)
    return {
        "spatial": spatial_metrics,
        "temporal": temporal_metrics,
        "association_accuracy": association_accuracy,
        "association_decisions": association_total,
        "selection_score": float(sum(available) / len(available)) if available else 0.0,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    overlap = {path.resolve() for path in args.train_features} & {
        path.resolve() for path in args.val_features
    }
    if overlap and not args.allow_overlapping_smoke_split:
        raise ValueError(
            "Training and validation feature shards overlap. This is forbidden for a reportable "
            "run; pass --allow-overlapping-smoke-split only for a local wiring test."
        )
    train_spatial, train_temporal, train_association, train_datasets = make_loaders(
        args.train_features, args.train_relations, args.batch_size, True
    )
    val_spatial, val_temporal, val_association, val_datasets = make_loaders(
        args.val_features, args.val_relations, args.batch_size, False
    )
    train_loaders = (train_spatial, train_temporal, train_association)
    val_loaders = (val_spatial, val_temporal, val_association)
    if all(loader is None for loader in train_loaders):
        raise RuntimeError("No Stage-2 training examples were found.")

    object_class_weight, object_class_counts = object_class_statistics(
        args.train_features,
        len(OBJECT_CLASSES),
        args.max_object_class_weight,
        args.full_multimodal_only,
    )
    spatial_pos_weight, spatial_positive_counts, spatial_examples = predicate_pos_weight(
        args.train_relations,
        "spatial",
        len(SPATIAL_PREDICATES),
        args.max_predicate_pos_weight,
    )
    temporal_pos_weight, temporal_positive_counts, temporal_examples = predicate_pos_weight(
        args.train_relations,
        "temporal",
        len(TEMPORAL_PREDICATES),
        args.max_predicate_pos_weight,
    )
    object_class_weight = object_class_weight.to(device)
    spatial_pos_weight = spatial_pos_weight.to(device)
    temporal_pos_weight = temporal_pos_weight.to(device)

    stage1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(**stage1.get("model_config", {}))
    validate_protocol(model_config)
    model = IBPK360Model(model_config).to(device)
    model.load_state_dict(stage1["model"], strict=True)
    encoder_ids = {id(parameter) for parameter in model.object_encoder.parameters()}
    encoder_parameters = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    graph_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": graph_parameters, "lr": args.learning_rate},
            {"params": encoder_parameters, "lr": args.learning_rate * args.encoder_lr_multiplier},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_score = -1.0
    start_epoch = 1
    if args.resume and args.last_checkpoint.is_file():
        checkpoint = torch.load(args.last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint.get("history", []))
        best_score = float(checkpoint.get("best_selection_score", -1.0))
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        iterators = [iter(loader) if loader is not None else None for loader in train_loaders]
        steps = args.max_train_steps or args.steps_per_epoch
        totals = {key: 0.0 for key in ("total", "spatial", "temporal", "association", "node", "representation", "consistency")}
        progress = trange(steps, desc=f"Stage 2 warm-up {epoch}/{args.epochs}")
        for step_index in progress:
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            spatial_batch, iterators[0] = next_or_none(iterators[0], train_spatial)
            temporal_batch, iterators[1] = next_or_none(iterators[1], train_temporal)
            association_batch, iterators[2] = next_or_none(iterators[2], train_association)
            if spatial_batch is not None:
                losses, _spatial_logits, _spatial_labels = spatial_step(
                    model,
                    spatial_batch,
                    device,
                    spatial_pos_weight,
                    object_class_weight,
                )
                representation = (
                    args.lambda_diversity * losses["diversity"]
                    + args.lambda_part * losses["part"]
                    + args.lambda_object * losses["object"]
                )
                total = total + args.lambda_spatial * losses["edge"] + args.lambda_node * losses["node"] + representation
                totals["spatial"] += float(losses["edge"].detach().cpu())
                totals["node"] += float(losses["node"].detach().cpu())
                totals["representation"] += float(representation.detach().cpu())
            if temporal_batch is not None:
                losses, _temporal_logits, _temporal_labels = temporal_step(
                    model, temporal_batch, device, temporal_pos_weight
                )
                total = total + args.lambda_temporal * losses["edge"] + args.lambda_consistency * losses["consistency"]
                totals["temporal"] += float(losses["edge"].detach().cpu())
                totals["consistency"] += float(losses["consistency"].detach().cpu())
            if association_batch is not None:
                association_loss, _association_correct, _association_total = association_step(
                    model, association_batch, device
                )
                total = total + args.lambda_association * association_loss
                totals["association"] += float(association_loss.detach().cpu())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["total"] += float(total.detach().cpu())
            progress.set_postfix(
                {key: round(value / (step_index + 1), 4) for key, value in totals.items()}
            )
        scheduler.step()
        validation = evaluate(
            model,
            val_loaders,
            device,
            args.max_val_batches,
            spatial_pos_weight,
            temporal_pos_weight,
            object_class_weight,
        )
        record = {
            "epoch": epoch,
            "train": {key: value / max(steps, 1) for key, value in totals.items()},
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        payload = {
            "schema_version": "IBP-K360-stage2-teacher-warmup-v1.0.0",
            "teacher_forced_tracklets": True,
            "final_predicted_tracklet_checkpoint": False,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_selection_score": max(best_score, validation["selection_score"]),
            "history": history,
            "model_config": model.config.to_dict(),
            "object_classes": list(OBJECT_CLASSES),
            "spatial_predicates": list(SPATIAL_PREDICATES),
            "temporal_predicates": list(TEMPORAL_PREDICATES),
            "training_statistics": {
                "object_class_counts": object_class_counts,
                "object_class_weights": object_class_weight.detach().cpu().tolist(),
                "spatial_examples": spatial_examples,
                "spatial_positive_counts": spatial_positive_counts,
                "spatial_pos_weight": spatial_pos_weight.detach().cpu().tolist(),
                "temporal_examples": temporal_examples,
                "temporal_positive_counts": temporal_positive_counts,
                "temporal_pos_weight": temporal_pos_weight.detach().cpu().tolist(),
            },
        }
        save_checkpoint(args.last_checkpoint, payload)
        if validation["selection_score"] > best_score:
            best_score = validation["selection_score"]
            save_checkpoint(args.best_checkpoint, payload)

    result = {
        "schema_version": "IBP-K360-stage2-teacher-warmup-result-v1.0.0",
        "teacher_forced_tracklets": True,
        "publishable_final_result": False,
        "enabled_modalities": list(model.config.enabled_modalities),
        "use_parts": model.config.use_parts,
        "best_validation_selection_score": best_score,
        "best_checkpoint": str(args.best_checkpoint.resolve()),
        "next_required_stage": "predicted-tracklet graph finetuning and held-out test evaluation",
        "selection_metric": "mean validation spatial mAP, temporal mAP, and bidirectional association accuracy",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset in itertools.chain(train_datasets, val_datasets):
        dataset.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KITTI-360 IBP Stage-2 teacher-forced warm-up.")
    parser.add_argument("--train-features", type=Path, nargs="+", required=True)
    parser.add_argument("--train-relations", type=Path, nargs="+", required=True)
    parser.add_argument("--val-features", type=Path, nargs="+", required=True)
    parser.add_argument("--val-relations", type=Path, nargs="+", required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("ibp_model/outputs/stage2_warmup"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1000,
        help="Balanced updates per epoch; each update draws one batch from every available task.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-diversity", type=float, default=0.1)
    parser.add_argument("--lambda-part", type=float, default=1.0)
    parser.add_argument("--lambda-object", type=float, default=1.0)
    parser.add_argument("--lambda-node", type=float, default=1.0)
    parser.add_argument("--lambda-spatial", type=float, default=1.0)
    parser.add_argument("--lambda-temporal", type=float, default=1.0)
    parser.add_argument("--lambda-association", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.2)
    parser.add_argument("--max-object-class-weight", type=float, default=5.0)
    parser.add_argument("--max-predicate-pos-weight", type=float, default=20.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-overlapping-smoke-split", action="store_true")
    parser.add_argument("--full-multimodal-only", action="store_true")
    args = parser.parse_args()
    if len(args.train_features) != len(args.train_relations):
        parser.error("Train feature and relation shard counts must match.")
    if len(args.val_features) != len(args.val_relations):
        parser.error("Validation feature and relation shard counts must match.")
    if args.full_multimodal_only:
        for path in [*args.train_relations, *args.val_relations]:
            with h5py.File(path.resolve(), "r") as relation_file:
                if not bool(relation_file.attrs.get("full_multimodal_only", False)):
                    parser.error(f"Relation shard is not full-multimodal: {path}")
    args.best_checkpoint = args.output_root / "checkpoints/stage2_warmup_best.pt"
    args.last_checkpoint = args.output_root / "checkpoints/stage2_warmup_last.pt"
    args.result = args.output_root / "stage2_warmup_result.json"
    return args


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2))


if __name__ == "__main__":
    main()
