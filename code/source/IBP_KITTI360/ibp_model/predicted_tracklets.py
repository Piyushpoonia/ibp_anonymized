from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from .model import IBPK360Model
from .protocol import ModelConfig, validate_protocol
from .relation_data import OBJECT_INPUT_KEYS


SCHEMA_VERSION = "IBP-K360-predicted-tracklets-v1.1.0"


class DisjointSet:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_feature_metadata(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any] | None] = [None] * expected_rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            index = int(value["feature_row"])
            if not 0 <= index < expected_rows or rows[index] is not None:
                raise ValueError(f"Invalid or duplicate feature row {index} in {path}")
            rows[index] = value
    if any(value is None for value in rows):
        raise ValueError(f"Feature metadata is incomplete: {path}")
    return [value for value in rows if value is not None]


def timeline_arrays(
    timeline_root: Path, metadata: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = pd.read_parquet(
        timeline_root / "tables/sample.parquet",
        columns=["token", "timestamp_ns", "sample_index", "raw_frame_index"],
    )
    lookup = {
        str(row.token): (int(row.timestamp_ns), int(row.sample_index))
        for row in samples.itertuples(index=False)
    }
    timestamps = np.empty(len(metadata), dtype=np.int64)
    sample_indices = np.empty(len(metadata), dtype=np.int64)
    for index, value in enumerate(metadata):
        sample_token = str(value["sample_token"])
        if sample_token not in lookup:
            raise ValueError(f"Feature sample does not resolve in the timeline: {sample_token}")
        timestamps[index], sample_indices[index] = lookup[sample_token]
    timeline_frames = samples["raw_frame_index"].to_numpy(dtype=np.int64)
    timeline_timestamps = samples["timestamp_ns"].to_numpy(dtype=np.int64)
    return timestamps, sample_indices, timeline_frames, timeline_timestamps


def load_model(checkpoint_path: Path, device: torch.device) -> IBPK360Model:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint.get("model_config", {}))
    validate_protocol(config)
    model = IBPK360Model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def feature_batch(
    features: h5py.File, rows: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    rows = np.asarray(rows, dtype=np.int64)

    def read_rows(dataset: h5py.Dataset) -> np.ndarray:
        # h5py requires increasing fancy indices. Preserve the candidate order
        # by sorting for the read and restoring the original order afterward.
        order = np.argsort(rows, kind="stable")
        sorted_values = np.asarray(dataset[rows[order]])
        return sorted_values[np.argsort(order)]

    result: dict[str, torch.Tensor] = {}
    for key in OBJECT_INPUT_KEYS:
        tensor = torch.from_numpy(read_rows(features[key]))
        tensor = tensor.bool() if key == "modality_mask" else tensor.float()
        result[key] = tensor.to(device, non_blocking=True)
    return result


def mutual_matches(
    augmented_logits: torch.Tensor, candidate_mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    ns, nt = candidate_mask.shape
    source_prediction = augmented_logits[:ns].argmax(dim=1).cpu().numpy()
    target_prediction = augmented_logits[:, :nt].argmax(dim=0).cpu().numpy()
    mask = candidate_mask.cpu().numpy().astype(bool)
    accepted: list[tuple[int, int]] = []
    for source, target in enumerate(source_prediction):
        target = int(target)
        if target >= nt:
            continue
        if int(target_prediction[target]) == source and mask[source, target]:
            accepted.append((source, target))
    return source_prediction, target_prediction, accepted


def compact_track_ids(
    disjoint: DisjointSet, eligible_rows: np.ndarray
) -> np.ndarray:
    result = np.full(len(disjoint.parent), -1, dtype=np.int64)
    roots = sorted({disjoint.find(int(row)) for row in eligible_rows})
    root_to_track = {root: index for index, root in enumerate(roots)}
    for row in eligible_rows:
        result[row] = root_to_track[disjoint.find(int(row))]
    return result


def prepare_predicted_tracklets(args: argparse.Namespace) -> dict[str, Any]:
    feature_path = args.feature_file.resolve()
    relation_path = args.relation_file.resolve()
    metadata_path = args.feature_metadata.resolve()
    timeline_root = args.timeline_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Predicted-tracklet output exists: {output_path}")
    device = torch.device(args.device)
    model = load_model(checkpoint_path, device)

    with h5py.File(feature_path, "r") as features:
        feature_rows = len(features["category_id"])
        metadata = read_feature_metadata(metadata_path, feature_rows)
        timestamps, sample_indices, timeline_frames, timeline_timestamps = timeline_arrays(
            timeline_root, metadata
        )
        modality_mask = np.asarray(features["modality_mask"], dtype=np.bool_)
        eligible_rows = np.flatnonzero(modality_mask.all(axis=1)).astype(np.int64)
        disjoint = DisjointSet(feature_rows)
        accepted_source: list[int] = []
        accepted_target: list[int] = []
        accepted_confidence: list[float] = []
        group_shapes: list[tuple[int, int]] = []
        group_source_rows: list[np.ndarray] = []
        group_target_rows: list[np.ndarray] = []
        group_logits: list[np.ndarray] = []
        group_candidate_masks: list[np.ndarray] = []
        group_source_predictions: list[np.ndarray] = []
        group_target_predictions: list[np.ndarray] = []
        source_correct = target_correct = source_total = target_total = 0
        true_links = predicted_links = correct_links = 0

        with h5py.File(relation_path, "r", swmr=True) as relations:
            if relations.attrs.get("schema_version", "") != "IBP-K360-relation-index-v1.2.0":
                raise ValueError("Stage 3 requires relation-index schema v1.2.0.")
            association = relations["association"]
            with torch.inference_mode():
                for index, shape in enumerate(association["shape"]):
                    ns, nt = (int(value) for value in shape)
                    source_rows = np.asarray(association["source_rows"][index], dtype=np.int64)
                    target_rows = np.asarray(association["target_rows"][index], dtype=np.int64)
                    geometry = torch.from_numpy(
                        np.asarray(association["geometry"][index], dtype=np.float32).reshape(ns, nt, 16)
                    ).to(device)
                    candidate_mask = torch.from_numpy(
                        np.asarray(association["candidate_mask"][index], dtype=np.uint8)
                        .reshape(ns, nt)
                        .astype(bool)
                    ).to(device)
                    source_encoded = model.encode_objects(
                        **feature_batch(features, source_rows, device)
                    )["fused_object"]
                    target_encoded = model.encode_objects(
                        **feature_batch(features, target_rows, device)
                    )["fused_object"]
                    output = model.association(
                        source_encoded, target_encoded, geometry, candidate_mask
                    )
                    source_prediction, target_prediction, accepted = mutual_matches(
                        output["augmented_logits"], candidate_mask
                    )
                    source_targets = np.asarray(
                        association["source_targets"][index], dtype=np.int64
                    )
                    target_targets = np.asarray(
                        association["target_targets"][index], dtype=np.int64
                    )
                    source_correct += int((source_prediction == source_targets).sum())
                    target_correct += int((target_prediction == target_targets).sum())
                    source_total += ns
                    target_total += nt
                    true_links += int((source_targets < nt).sum())
                    predicted_links += len(accepted)

                    logits = output["augmented_logits"].detach().cpu().numpy().astype(np.float32)
                    for source, target in accepted:
                        source_row = int(source_rows[source])
                        target_row = int(target_rows[target])
                        if metadata[source_row]["scene_token"] != metadata[target_row]["scene_token"]:
                            raise ValueError("A predicted association crossed a scene boundary.")
                        if sample_indices[target_row] != sample_indices[source_row] + 1:
                            raise ValueError("Association candidates are not adjacent key samples.")
                        disjoint.union(source_row, target_row)
                        accepted_source.append(source_row)
                        accepted_target.append(target_row)
                        accepted_confidence.append(float(torch.sigmoid(output["pair_logits"][source, target]).cpu()))
                        correct_links += int(source_targets[source] == target)

                    group_shapes.append((ns, nt))
                    group_source_rows.append(source_rows)
                    group_target_rows.append(target_rows)
                    group_logits.append(logits.reshape(-1))
                    group_candidate_masks.append(candidate_mask.cpu().numpy().astype(np.uint8).reshape(-1))
                    group_source_predictions.append(source_prediction.astype(np.int64))
                    group_target_predictions.append(target_prediction.astype(np.int64))

        track_ids = compact_track_ids(disjoint, eligible_rows)
        track_members: dict[int, list[int]] = defaultdict(list)
        for row in eligible_rows:
            track_members[int(track_ids[row])].append(int(row))
        duplicate_frames = 0
        purities: list[float] = []
        for members in track_members.values():
            frame_keys = [
                (str(metadata[row]["scene_token"]), int(sample_indices[row])) for row in members
            ]
            duplicate_frames += len(frame_keys) - len(set(frame_keys))
            identities = [str(metadata[row]["instance_token_supervision_only"]) for row in members]
            purities.append(max(Counter(identities).values()) / len(identities))
        if duplicate_frames:
            raise ValueError("Predicted tracks contain multiple objects in one sample.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        variable_int = h5py.vlen_dtype(np.dtype("int64"))
        variable_float = h5py.vlen_dtype(np.dtype("float32"))
        variable_byte = h5py.vlen_dtype(np.dtype("uint8"))
        with h5py.File(temporary, "w") as target:
            target.attrs["schema_version"] = SCHEMA_VERSION
            target.attrs["source_checkpoint_sha256"] = sha256_file(checkpoint_path)
            target.attrs["sequence"] = str(metadata[0]["sequence"])
            target.create_dataset("row_track_ids", data=track_ids, compression="gzip")
            target.create_dataset("row_timestamps_ns", data=timestamps, compression="gzip")
            target.create_dataset("row_sample_indices", data=sample_indices, compression="gzip")
            target.create_dataset(
                "timeline_raw_frame_indices", data=timeline_frames, compression="gzip"
            )
            target.create_dataset(
                "timeline_timestamps_ns", data=timeline_timestamps, compression="gzip"
            )
            target.create_dataset(
                "row_raw_frame_indices",
                data=np.asarray([int(value["raw_frame_index"]) for value in metadata]),
                compression="gzip",
            )
            target.create_dataset(
                "row_annotation_tokens",
                data=np.asarray([value["annotation_token"] for value in metadata], dtype="S32"),
                compression="gzip",
            )
            target.create_dataset(
                "row_sample_tokens",
                data=np.asarray([value["sample_token"] for value in metadata], dtype="S32"),
                compression="gzip",
            )
            target.create_dataset(
                "row_scene_tokens",
                data=np.asarray([value["scene_token"] for value in metadata], dtype="S32"),
                compression="gzip",
            )
            links = target.create_group("accepted_links")
            links.create_dataset("source_rows", data=np.asarray(accepted_source, dtype=np.int64))
            links.create_dataset("target_rows", data=np.asarray(accepted_target, dtype=np.int64))
            links.create_dataset("confidence", data=np.asarray(accepted_confidence, dtype=np.float32))
            groups = target.create_group("association_groups")
            groups.create_dataset("shape", data=np.asarray(group_shapes, dtype=np.int32))
            for name, dtype, values in (
                ("source_rows", variable_int, group_source_rows),
                ("target_rows", variable_int, group_target_rows),
                ("raw_augmented_logits", variable_float, group_logits),
                ("candidate_mask", variable_byte, group_candidate_masks),
                ("source_predictions", variable_int, group_source_predictions),
                ("target_predictions", variable_int, group_target_predictions),
            ):
                dataset = groups.create_dataset(name, (len(values),), dtype=dtype)
                for index, value in enumerate(values):
                    dataset[index] = value
        temporary.replace(output_path)

    association_decisions = source_total + target_total
    report = {
        "schema_version": SCHEMA_VERSION,
        "sequence": str(metadata[0]["sequence"]),
        "feature_rows": feature_rows,
        "full_multimodal_rows": int(len(eligible_rows)),
        "association_groups": len(group_shapes),
        "association_decisions": association_decisions,
        "association_correct": source_correct + target_correct,
        "association_accuracy": (source_correct + target_correct) / max(association_decisions, 1),
        "true_forward_links": true_links,
        "predicted_mutual_links": predicted_links,
        "correct_mutual_links": correct_links,
        "mutual_link_precision": correct_links / max(predicted_links, 1),
        "mutual_link_recall": correct_links / max(true_links, 1),
        "predicted_tracks": len(track_members),
        "mean_track_length": (
            float(np.mean([len(value) for value in track_members.values()]))
            if track_members
            else 0.0
        ),
        "maximum_track_length": max((len(value) for value in track_members.values()), default=0),
        "identity_purity_evaluation_only": float(np.mean(purities)) if purities else 0.0,
        "identity_privacy": (
            "Persistent identities were used only for post-hoc purity and target metrics, "
            "never for association prediction or track construction."
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output": str(output_path),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KITTI-360 tracklets using predicted dustbin-aware associations."
    )
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--relation-file", type=Path, required=True)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare_predicted_tracklets(parse_args()), indent=2))


if __name__ == "__main__":
    main()
