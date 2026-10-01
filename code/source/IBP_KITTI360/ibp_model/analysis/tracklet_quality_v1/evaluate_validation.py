from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...evaluate_stage2_predcls import validate_checkpoint_metadata
from ...evaluate_stage3 import relation_metadata, write_prediction_group
from ...metrics import multilabel_metrics
from ...model import IBPK360Model
from ...predicted_relation_data import PredictedTemporalRelationDataset
from ...predicted_tracklets import sha256_file
from ...protocol import ModelConfig, TEMPORAL_PREDICATES, validate_protocol
from ...relation_data import TemporalRelationDataset
from ...train_stage2_warmup import temporal_step as teacher_temporal_step
from ...train_stage3_predicted import (
    temporal_step as predicted_temporal_step,
    validate_track_sources,
)


SCHEMA_VERSION = "IBP-K360-tracklet-quality-validation-predictions-v1.0.0"


def load_split(path: Path, relation_paths: list[Path]) -> list[str]:
    split = json.loads(path.resolve().read_text(encoding="utf-8"))
    if split.get("schema_version") != "IBP-K360-sequence-split-v1.0.0":
        raise ValueError("Unexpected split schema")
    expected = [str(value) for value in split.get("validation", [])]
    actual = [path.stem for path in relation_paths]
    if expected != actual:
        raise ValueError(
            f"Validation shards do not match the frozen split: expected={expected}, actual={actual}"
        )
    return expected


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    checkpoint_validator: Callable[[dict[str, Any]], None],
) -> tuple[IBPK360Model, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=False
    )
    checkpoint_validator(checkpoint)
    config = ModelConfig(**checkpoint["model_config"])
    validate_protocol(config)
    model = IBPK360Model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def validate_stage3_checkpoint(checkpoint: dict[str, Any]) -> None:
    if not bool(checkpoint.get("final_predicted_tracklet_checkpoint", False)):
        raise ValueError("Predicted validation requires the frozen Stage 3 checkpoint")
    if bool(checkpoint.get("teacher_forced_tracklets", True)):
        raise ValueError("Stage 3 checkpoint is incorrectly marked teacher-forced")
    if not checkpoint.get("association_source_checkpoint_sha256"):
        raise ValueError("Stage 3 checkpoint lacks its association source hash")


def release_model(model: IBPK360Model) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def teacher_predictions(
    feature_paths: list[Path],
    relation_paths: list[Path],
    checkpoint_path: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    model, checkpoint = load_model(
        checkpoint_path, device, validate_checkpoint_metadata
    )
    dataset = TemporalRelationDataset(feature_paths, relation_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    try:
        for batch in tqdm(loader, desc="Validation teacher-forced temporal"):
            _, batch_logits, batch_targets = teacher_temporal_step(
                model, batch, device, None
            )
            logits.append(batch_logits.cpu())
            targets.append(batch_targets.cpu())
    finally:
        dataset.close()
    if not logits:
        raise ValueError("Teacher-forced validation produced no temporal predictions")
    result_logits = torch.cat(logits).numpy().astype(np.float32)
    result_targets = torch.cat(targets).numpy().astype(np.uint8)
    metrics = multilabel_metrics(
        torch.from_numpy(result_logits),
        torch.from_numpy(result_targets),
        TEMPORAL_PREDICATES,
    )
    release_model(model)
    return result_logits, result_targets, metrics, checkpoint


@torch.inference_mode()
def predicted_predictions(
    feature_paths: list[Path],
    relation_paths: list[Path],
    track_paths: list[Path],
    checkpoint_path: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    model, checkpoint = load_model(
        checkpoint_path, device, validate_stage3_checkpoint
    )
    validate_track_sources(
        track_paths, str(checkpoint["association_source_checkpoint_sha256"])
    )
    dataset = PredictedTemporalRelationDataset(
        feature_paths, relation_paths, track_paths
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    available: list[torch.Tensor] = []
    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    joint_masks: list[torch.Tensor] = []
    try:
        for batch in tqdm(loader, desc="Validation predicted-track temporal"):
            _, batch_logits, batch_targets = predicted_temporal_step(
                model, batch, device, None
            )
            batch_available = batch["available"].bool()
            batch_logits = batch_logits.cpu()
            batch_logits[~batch_available] = -20.0
            logits.append(batch_logits)
            targets.append(batch_targets.cpu())
            available.append(batch_available)
            source_rows.append(batch["predicted_source_rows"].long())
            target_rows.append(batch["predicted_target_rows"].long())
            joint_masks.append(batch["predicted_joint_mask"].bool())
    finally:
        dataset.close()
    if not logits:
        raise ValueError("Predicted-track validation produced no temporal predictions")
    result_logits = torch.cat(logits).numpy().astype(np.float32)
    result_targets = torch.cat(targets).numpy().astype(np.uint8)
    extras = {
        "tracklet_available": torch.cat(available).numpy().astype(np.uint8),
        "predicted_source_rows": torch.cat(source_rows).numpy().astype(np.int64),
        "predicted_target_rows": torch.cat(target_rows).numpy().astype(np.int64),
        "predicted_joint_mask": torch.cat(joint_masks).numpy().astype(np.uint8),
    }
    metrics = multilabel_metrics(
        torch.from_numpy(result_logits),
        torch.from_numpy(result_targets),
        TEMPORAL_PREDICATES,
    )
    metrics["predicted_tracklet_coverage"] = float(extras["tracklet_available"].mean())
    release_model(model)
    return result_logits, result_targets, extras, metrics, checkpoint


def write_predictions(
    path: Path,
    logits: np.ndarray,
    targets: np.ndarray,
    metadata: dict[str, Any],
    checkpoint_path: Path,
    sequences: list[str],
    **extras: np.ndarray,
) -> None:
    temporary = path.with_suffix(".h5.tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["split"] = "validation"
        output.attrs["sequences"] = json.dumps(sequences)
        output.attrs["checkpoint_sha256"] = sha256_file(checkpoint_path.resolve())
        write_prediction_group(
            output.create_group("temporal"),
            logits,
            targets,
            metadata["relation_ids"],
            metadata["group_ids"],
            metadata["sequences"],
            TEMPORAL_PREDICATES,
            source_annotation_tokens=metadata["source_annotation_tokens"],
            target_annotation_tokens=metadata["target_annotation_tokens"],
            raw_frame_indices=metadata["raw_frame_indices"],
            **extras,
        )
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "validation_prediction_manifest.json"
    teacher_path = output_root / "teacher_raw_predictions.h5"
    predicted_path = output_root / "predicted_raw_predictions.h5"
    if (
        manifest_path.exists() or teacher_path.exists() or predicted_path.exists()
    ) and not args.force:
        raise FileExistsError(
            f"Validation prediction output already exists in {output_root}"
        )
    sizes = {
        len(args.feature_files),
        len(args.relation_files),
        len(args.track_files),
    }
    if len(sizes) != 1:
        raise ValueError("Validation feature, relation, and track files must align")
    sequences = load_split(args.split_file, args.relation_files)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    metadata = relation_metadata(args.relation_files, "temporal")

    teacher_logits, teacher_targets, teacher_metrics, teacher_checkpoint = (
        teacher_predictions(
            args.feature_files,
            args.relation_files,
            args.stage2_checkpoint,
            args.batch_size,
            args.workers,
            device,
        )
    )
    predicted_logits, predicted_targets, extras, predicted_metrics, stage3_checkpoint = (
        predicted_predictions(
            args.feature_files,
            args.relation_files,
            args.track_files,
            args.stage3_checkpoint,
            args.batch_size,
            args.workers,
            device,
        )
    )
    if not np.array_equal(teacher_targets, predicted_targets):
        raise ValueError("Teacher and predicted validation targets differ")
    if len(metadata["relation_ids"]) != len(teacher_logits):
        raise ValueError("Validation relation metadata and logits are misaligned")
    if teacher_checkpoint["model_config"] != stage3_checkpoint["model_config"]:
        raise ValueError("Stage 2 and Stage 3 model configurations differ")

    write_predictions(
        teacher_path,
        teacher_logits,
        teacher_targets,
        metadata,
        args.stage2_checkpoint,
        sequences,
    )
    write_predictions(
        predicted_path,
        predicted_logits,
        predicted_targets,
        metadata,
        args.stage3_checkpoint,
        sequences,
        **extras,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "split": "validation",
        "sequences": sequences,
        "teacher_forced": teacher_metrics,
        "predicted_tracklet": predicted_metrics,
        "teacher_checkpoint": str(args.stage2_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(args.stage2_checkpoint.resolve()),
        "predicted_checkpoint": str(args.stage3_checkpoint.resolve()),
        "predicted_checkpoint_sha256": sha256_file(args.stage3_checkpoint.resolve()),
        "teacher_predictions": str(teacher_path),
        "predicted_predictions": str(predicted_path),
        "publishable_final_result": False,
        "purpose": "Validation-only tracklet-quality diagnosis and method development.",
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired Stage 2 and Stage 3 temporal predictions on validation."
    )
    parser.add_argument("--feature-files", type=Path, nargs="+", required=True)
    parser.add_argument("--relation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--track-files", type=Path, nargs="+", required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
