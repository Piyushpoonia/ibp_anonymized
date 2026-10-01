from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate_stage3 import (
    predicate_recall_at_k,
    relation_metadata,
    write_prediction_group,
)
from .metrics import multilabel_metrics
from .model import IBPK360Model
from .predicted_tracklets import sha256_file
from .protocol import (
    ModelConfig,
    OBJECT_CLASSES,
    SPATIAL_PREDICATES,
    TEMPORAL_PREDICATES,
    validate_protocol,
)
from .relation_data import SpatialRelationDataset, TemporalRelationDataset
from .train_stage2_warmup import spatial_step, temporal_step


SCHEMA_VERSION = "IBP-K360-stage2-predcls-held-out-test-v1.0.0"


def validate_checkpoint_metadata(checkpoint: dict[str, Any]) -> None:
    if not bool(checkpoint.get("teacher_forced_tracklets", False)):
        raise ValueError("PredCLS evaluation requires a teacher-forced Stage 2 checkpoint.")
    if bool(checkpoint.get("final_predicted_tracklet_checkpoint", False)):
        raise ValueError("A predicted-tracklet Stage 3 checkpoint cannot be used for PredCLS.")
    if tuple(checkpoint.get("object_classes", ())) != OBJECT_CLASSES:
        raise ValueError("Checkpoint object-class order differs from the frozen protocol.")
    if tuple(checkpoint.get("spatial_predicates", ())) != SPATIAL_PREDICATES:
        raise ValueError("Checkpoint spatial-predicate order differs from the frozen protocol.")
    if tuple(checkpoint.get("temporal_predicates", ())) != TEMPORAL_PREDICATES:
        raise ValueError("Checkpoint temporal-predicate order differs from the frozen protocol.")
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint is missing model weights or model configuration.")


def load_and_validate_split(
    split_path: Path,
    feature_paths: list[Path],
    relation_paths: list[Path],
) -> tuple[dict[str, Any], list[str]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema_version") != "IBP-K360-sequence-split-v1.0.0":
        raise ValueError("Unexpected KITTI-360 split schema.")
    if split.get("split_unit") != "complete driving sequence":
        raise ValueError("Held-out evaluation requires a complete-sequence split.")
    expected = [str(value) for value in split.get("test", [])]
    if not expected:
        raise ValueError("The frozen split contains no held-out test sequences.")
    feature_sequences = [path.stem for path in feature_paths]
    relation_sequences = [path.stem for path in relation_paths]
    if feature_sequences != expected or relation_sequences != expected:
        raise ValueError(
            "Test shards do not match the frozen held-out sequence order: "
            f"expected={expected}, features={feature_sequences}, relations={relation_sequences}"
        )
    return split, expected


def runtime_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def ensure_output_below(output_root: Path, project_root: Path) -> Path:
    resolved_output = output_root.resolve()
    resolved_project = project_root.resolve()
    if not resolved_output.is_relative_to(resolved_project):
        raise ValueError(f"Output must remain below the project root: {resolved_output}")
    return resolved_output


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_final_test != "YES":
        raise ValueError("Held-out PredCLS evaluation is locked; pass --run-final-test YES.")
    if len(args.test_features) != len(args.test_relations):
        raise ValueError("Feature and relation test shards must be aligned.")

    output_root = ensure_output_below(args.output_root, args.project_root)
    metrics_path = output_root / "final_test_metrics.json"
    prediction_path = output_root / "raw_predictions.h5"
    if metrics_path.exists() or prediction_path.exists():
        raise FileExistsError(
            f"Stage 2 PredCLS held-out output already exists in {output_root}; refusing a repeat."
        )

    split, test_sequences = load_and_validate_split(
        args.split_file.resolve(),
        [path.resolve() for path in args.test_features],
        [path.resolve() for path in args.test_relations],
    )
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(checkpoint)
    model_config = ModelConfig(**checkpoint["model_config"])
    validate_protocol(model_config)

    device = torch.device(args.device)
    model = IBPK360Model(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    spatial_dataset = SpatialRelationDataset(args.test_features, args.test_relations)
    temporal_dataset = TemporalRelationDataset(args.test_features, args.test_relations)
    spatial_loader = DataLoader(
        spatial_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    temporal_loader = DataLoader(
        temporal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    spatial_logits: list[torch.Tensor] = []
    spatial_labels: list[torch.Tensor] = []
    temporal_logits: list[torch.Tensor] = []
    temporal_labels: list[torch.Tensor] = []
    try:
        for batch in tqdm(spatial_loader, desc="Stage 2 PredCLS spatial test"):
            _, logits, labels = spatial_step(model, batch, device)
            spatial_logits.append(logits.cpu())
            spatial_labels.append(labels.cpu())
        for batch in tqdm(temporal_loader, desc="Stage 2 PredCLS temporal test"):
            _, logits, labels = temporal_step(model, batch, device)
            temporal_logits.append(logits.cpu())
            temporal_labels.append(labels.cpu())
    finally:
        spatial_dataset.close()
        temporal_dataset.close()

    if not spatial_logits or not temporal_logits:
        raise ValueError("Held-out PredCLS evaluation produced an empty task output.")
    spatial_logits_array = torch.cat(spatial_logits).numpy().astype(np.float32)
    spatial_labels_array = torch.cat(spatial_labels).numpy().astype(np.uint8)
    temporal_logits_array = torch.cat(temporal_logits).numpy().astype(np.float32)
    temporal_labels_array = torch.cat(temporal_labels).numpy().astype(np.uint8)

    spatial_metadata = relation_metadata(args.test_relations, "spatial")
    temporal_metadata = relation_metadata(args.test_relations, "temporal")
    if len(spatial_metadata["relation_ids"]) != len(spatial_logits_array):
        raise ValueError("Spatial relation metadata and logits are misaligned.")
    if len(temporal_metadata["relation_ids"]) != len(temporal_logits_array):
        raise ValueError("Temporal relation metadata and logits are misaligned.")

    spatial_metrics = multilabel_metrics(
        torch.from_numpy(spatial_logits_array),
        torch.from_numpy(spatial_labels_array),
        SPATIAL_PREDICATES,
    )
    temporal_metrics = multilabel_metrics(
        torch.from_numpy(temporal_logits_array),
        torch.from_numpy(temporal_labels_array),
        TEMPORAL_PREDICATES,
    )
    spatial_ranking = predicate_recall_at_k(
        spatial_logits_array,
        spatial_labels_array,
        spatial_metadata["group_ids"],
        SPATIAL_PREDICATES,
    )
    temporal_ranking = predicate_recall_at_k(
        temporal_logits_array,
        temporal_labels_array,
        temporal_metadata["group_ids"],
        TEMPORAL_PREDICATES,
    )

    checkpoint_hash = sha256_file(checkpoint_path)
    split_hash = sha256_file(args.split_file.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_temporary = prediction_path.with_suffix(".h5.tmp")
    with h5py.File(prediction_temporary, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["model_name"] = "IBP-K360 Stage 2 teacher-forced (PredCLS)"
        output.attrs["split"] = "held_out_test"
        output.attrs["test_sequences"] = json.dumps(test_sequences)
        output.attrs["checkpoint_sha256"] = checkpoint_hash
        output.attrs["split_file_sha256"] = split_hash
        output.attrs["threshold"] = 0.5
        write_prediction_group(
            output.create_group("spatial"),
            spatial_logits_array,
            spatial_labels_array,
            spatial_metadata["relation_ids"],
            spatial_metadata["group_ids"],
            spatial_metadata["sequences"],
            SPATIAL_PREDICATES,
            source_annotation_tokens=spatial_metadata["source_annotation_tokens"],
            target_annotation_tokens=spatial_metadata["target_annotation_tokens"],
            raw_frame_indices=spatial_metadata["raw_frame_indices"],
        )
        write_prediction_group(
            output.create_group("temporal"),
            temporal_logits_array,
            temporal_labels_array,
            temporal_metadata["relation_ids"],
            temporal_metadata["group_ids"],
            temporal_metadata["sequences"],
            TEMPORAL_PREDICATES,
            source_annotation_tokens=temporal_metadata["source_annotation_tokens"],
            target_annotation_tokens=temporal_metadata["target_annotation_tokens"],
            raw_frame_indices=temporal_metadata["raw_frame_indices"],
        )
    prediction_temporary.replace(prediction_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "model_name": "IBP-K360 Stage 2 teacher-forced (PredCLS)",
        "comparison_setting": "ground-truth relation pairs and teacher-forced tracklets",
        "teacher_forced_tracklets": True,
        "held_out_test_evaluated": True,
        "split": "held_out_test",
        "test_sequences": test_sequences,
        "seed": int(split.get("training_seed", 42)),
        "threshold": 0.5,
        "object_classes": list(OBJECT_CLASSES),
        "spatial_predicates": list(SPATIAL_PREDICATES),
        "temporal_predicates": list(TEMPORAL_PREDICATES),
        "enabled_modalities": list(model_config.enabled_modalities),
        "use_parts": model_config.use_parts,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_sha256": checkpoint_hash,
        "split_file_sha256": split_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "spatial": spatial_metrics,
        "temporal": temporal_metrics,
        "spatial_triplet_ranking": spatial_ranking,
        "temporal_triplet_ranking": temporal_ranking,
        "runtime": runtime_record(),
        "raw_predictions": str(prediction_path),
    }
    metrics_temporary = metrics_path.with_suffix(".json.tmp")
    metrics_temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics_temporary.replace(metrics_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time IBP Stage 2 teacher-forced PredCLS held-out evaluation."
    )
    parser.add_argument("--run-final-test", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, nargs="+", required=True)
    parser.add_argument("--test-relations", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
