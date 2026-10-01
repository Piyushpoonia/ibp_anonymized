from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .feature_data import FeatureShardDataset
from .losses import representation_losses
from .model import IBPK360Model
from .protocol import ModelConfig, OBJECT_CLASSES, SPATIAL_PREDICATES, TEMPORAL_PREDICATES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def object_class_statistics(
    feature_paths: list[Path],
    classes: int,
    maximum_weight: float,
    full_multimodal_only: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Compute class weights from training shards only."""
    counts = np.zeros(classes, dtype=np.int64)
    for path in feature_paths:
        with h5py.File(path.resolve(), "r") as feature_file:
            labels = np.asarray(feature_file["category_id"], dtype=np.int64)
            if full_multimodal_only:
                masks = np.asarray(feature_file["modality_mask"], dtype=np.bool_)
                labels = labels[masks.all(axis=1)]
        counts += np.bincount(labels, minlength=classes)[:classes]
    if not counts.sum():
        raise RuntimeError("Training feature shards contain no object labels.")
    weights = np.zeros(classes, dtype=np.float32)
    supported = counts > 0
    weights[supported] = np.sqrt(counts.sum() / (supported.sum() * counts[supported]))
    weights[supported] /= weights[supported].mean()
    weights = np.clip(weights, 0.0, maximum_weight)
    return torch.from_numpy(weights), counts.tolist()


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in (
            "rgb_tokens",
            "lidar_tokens",
            "lidar_anchor",
            "text_embedding",
            "modality_mask",
        )
    }


def macro_f1(predictions: torch.Tensor, targets: torch.Tensor, classes: int) -> float:
    scores = []
    for class_id in range(classes):
        predicted = predictions.eq(class_id)
        actual = targets.eq(class_id)
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        denominator = 2 * tp + fp + fn
        if actual.any():
            scores.append(2 * tp / denominator if denominator else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


@torch.inference_mode()
def evaluate(
    model: IBPK360Model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    losses = []
    predictions = []
    targets = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        outputs = model.encode_objects(**move_inputs(batch, device))
        labels = batch["category_id"].to(device)
        logits = model.predict_nodes(outputs["fused_object"])
        losses.append(F.cross_entropy(logits, labels).cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(labels.cpu())
    if not targets:
        raise RuntimeError("Validation loader produced no batches.")
    predicted = torch.cat(predictions)
    actual = torch.cat(targets)
    return {
        "loss": float(torch.stack(losses).mean()),
        "accuracy": float(predicted.eq(actual).float().mean()),
        "macro_f1": macro_f1(predicted, actual, len(OBJECT_CLASSES)),
        "objects": int(actual.numel()),
    }


def save_checkpoint(
    path: Path,
    model: IBPK360Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_macro_f1: float,
    history: list[dict[str, object]],
    args: argparse.Namespace,
    object_counts: list[int],
    object_weights: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "IBP-K360-stage1-checkpoint-v1.0.0",
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation_macro_f1": best_macro_f1,
            "history": history,
            "model_config": model.config.to_dict(),
            "object_classes": list(OBJECT_CLASSES),
            "spatial_predicates": list(SPATIAL_PREDICATES),
            "temporal_predicates": list(TEMPORAL_PREDICATES),
            "arguments": vars(args),
            "training_object_class_counts": object_counts,
            "training_object_class_weights": object_weights.detach().cpu().tolist(),
        },
        temporary,
    )
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.smoke and args.train_features != args.val_features:
        raise ValueError("Smoke mode expects the same tiny shard for train and validation.")
    overlap = {path.resolve() for path in args.train_features} & {
        path.resolve() for path in args.val_features
    }
    if overlap and not args.smoke:
        raise ValueError(
            "Training and validation feature shards overlap. Use disjoint sequence shards; "
            "--smoke is the only mode that permits overlap."
        )
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_data = FeatureShardDataset(
        args.train_features, require_all_modalities=args.full_multimodal_only
    )
    val_data = FeatureShardDataset(
        args.val_features, require_all_modalities=args.full_multimodal_only
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_data) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = IBPK360Model(
        ModelConfig(
            enabled_modalities=tuple(args.enabled_modalities),
            use_parts=not args.whole_object,
        )
    ).to(device)
    object_weights, object_counts = object_class_statistics(
        args.train_features,
        len(OBJECT_CLASSES),
        args.max_object_class_weight,
        args.full_multimodal_only,
    )
    object_weights = object_weights.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 1
    best_macro_f1 = -1.0
    history: list[dict[str, object]] = []

    if args.resume and args.last_checkpoint.is_file():
        checkpoint = torch.load(args.last_checkpoint, map_location=device, weights_only=False)
        saved_config = ModelConfig(**checkpoint.get("model_config", {}))
        if (
            saved_config.enabled_modalities != model.config.enabled_modalities
            or saved_config.use_parts != model.config.use_parts
        ):
            raise ValueError(
                "Resume checkpoint representation configuration does not match this run: "
                f"modalities={saved_config.enabled_modalities}, use_parts={saved_config.use_parts} "
                f"!= modalities={model.config.enabled_modalities}, "
                f"use_parts={model.config.use_parts}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint["best_validation_macro_f1"])
        history = list(checkpoint.get("history", []))
        print(f"Resumed Stage 1 from epoch {start_epoch - 1}.")

    final_epoch_this_run = args.epochs
    if args.epochs_per_run is not None:
        final_epoch_this_run = min(
            args.epochs,
            start_epoch + args.epochs_per_run - 1,
        )

    for epoch in range(start_epoch, final_epoch_this_run + 1):
        model.train()
        totals = {"total": 0.0, "node": 0.0, "diversity": 0.0, "part": 0.0, "object": 0.0}
        batches = 0
        progress = tqdm(train_loader, desc=f"Stage 1 {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            outputs = model.encode_objects(**move_inputs(batch, device))
            losses = representation_losses(outputs)
            labels = batch["category_id"].to(device)
            node_loss = F.cross_entropy(
                model.predict_nodes(outputs["fused_object"]), labels, weight=object_weights
            )
            total = (
                args.lambda_diversity * losses["diversity"]
                + args.lambda_part * losses["part_alignment"]
                + args.lambda_object * losses["object_alignment"]
                + args.lambda_node * node_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batches += 1
            values = {
                "total": total,
                "node": node_loss,
                "diversity": losses["diversity"],
                "part": losses["part_alignment"],
                "object": losses["object_alignment"],
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu())
            progress.set_postfix({key: round(value / batches, 4) for key, value in totals.items()})
        scheduler.step()
        validation = evaluate(model, val_loader, device, args.max_val_batches)
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        save_checkpoint(
            args.last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            max(best_macro_f1, validation["macro_f1"]),
            history,
            args,
            object_counts,
            object_weights,
        )
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            save_checkpoint(
                args.best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                best_macro_f1,
                history,
                args,
                object_counts,
                object_weights,
            )

    result = {
        "schema_version": "IBP-K360-stage1-result-v1.0.0",
        "smoke_test_only": args.smoke,
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint": str(args.best_checkpoint.resolve()),
        "last_checkpoint": str(args.last_checkpoint.resolve()),
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "target_epochs": args.epochs,
        "training_complete": bool(history and history[-1]["epoch"] >= args.epochs),
        "train_objects": len(train_data),
        "validation_objects": len(val_data),
        "training_object_class_counts": object_counts,
        "training_object_class_weights": object_weights.detach().cpu().tolist(),
        "enabled_modalities": list(model.config.enabled_modalities),
        "use_parts": model.config.use_parts,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    train_data.close()
    val_data.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 of the KITTI-360 IBP model.")
    parser.add_argument("--train-features", type=Path, nargs="+", required=True)
    parser.add_argument("--val-features", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("ibp_model/outputs/stage1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--epochs-per-run",
        type=int,
        help="Optional wall-time chunk; resume jobs continue toward --epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-diversity", type=float, default=0.1)
    parser.add_argument("--lambda-part", type=float, default=1.0)
    parser.add_argument("--lambda-object", type=float, default=1.0)
    parser.add_argument("--lambda-node", type=float, default=1.0)
    parser.add_argument("--max-object-class-weight", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--full-multimodal-only",
        action="store_true",
        help="Train only on objects with RGB, LiDAR and caption-text features.",
    )
    parser.add_argument(
        "--enabled-modalities",
        nargs="+",
        choices=("rgb", "lidar", "text"),
        default=["rgb", "lidar", "text"],
        help="Sensor inputs available to the model; pairwise geometry is unchanged.",
    )
    parser.add_argument(
        "--whole-object",
        action="store_true",
        help=(
            "Disable latent part decomposition, part alignment, and part-pair "
            "interaction while retaining whole-object multimodal fusion."
        ),
    )
    args = parser.parse_args()
    if args.epochs_per_run is not None and args.epochs_per_run < 1:
        parser.error("--epochs-per-run must be at least 1.")
    args.best_checkpoint = args.output_root / "checkpoints/stage1_best_macro_f1.pt"
    args.last_checkpoint = args.output_root / "checkpoints/stage1_last.pt"
    args.result = args.output_root / "stage1_result.json"
    return args


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
