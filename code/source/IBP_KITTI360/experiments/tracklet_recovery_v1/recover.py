from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ...ibp_model.predicted_tracklets import (
    DisjointSet,
    compact_track_ids,
    sha256_file,
)
from . import RECOVERY_SCHEMA_VERSION


@dataclass(frozen=True)
class Link:
    source_index: int
    target_index: int
    pair_logit: float
    margin: float
    fallback: bool


def load_association_targets(path: Path) -> list[dict[str, np.ndarray | tuple[int, int]]]:
    groups: list[dict[str, np.ndarray | tuple[int, int]]] = []
    with h5py.File(path.resolve(), "r", swmr=True) as relations:
        if str(relations.attrs.get("schema_version", "")) != "IBP-K360-relation-index-v1.2.0":
            raise ValueError(f"Unexpected relation-index schema: {path}")
        association = relations["association"]
        for index, shape in enumerate(association["shape"]):
            ns, nt = (int(value) for value in shape)
            groups.append(
                {
                    "shape": (ns, nt),
                    "source_rows": np.asarray(
                        association["source_rows"][index], dtype=np.int64
                    ),
                    "target_rows": np.asarray(
                        association["target_rows"][index], dtype=np.int64
                    ),
                    "source_targets": np.asarray(
                        association["source_targets"][index], dtype=np.int64
                    ),
                    "target_targets": np.asarray(
                        association["target_targets"][index], dtype=np.int64
                    ),
                }
            )
    return groups


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


def select_links(
    augmented_logits: np.ndarray,
    candidate_mask: np.ndarray,
    mode: str,
    margin_threshold: float,
) -> list[Link]:
    ns, nt = candidate_mask.shape
    if augmented_logits.shape != (ns + 1, nt + 1):
        raise ValueError("Augmented association logits have an unexpected shape")
    if mode not in {"one_way", "all"}:
        raise ValueError(f"Unsupported recovery mode: {mode}")

    source_prediction = augmented_logits[:ns].argmax(axis=1)
    target_prediction = augmented_logits[:, :nt].argmax(axis=0)
    accepted: list[Link] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()

    for source, target_value in enumerate(source_prediction):
        target = int(target_value)
        if target >= nt or not bool(candidate_mask[source, target]):
            continue
        if int(target_prediction[target]) != source:
            continue
        pair_logit = float(augmented_logits[source, target])
        margin = min(
            pair_logit - float(augmented_logits[source, nt]),
            pair_logit - float(augmented_logits[ns, target]),
        )
        accepted.append(Link(source, target, pair_logit, margin, False))
        used_sources.add(source)
        used_targets.add(target)

    candidates: list[Link] = []
    for source in range(ns):
        if source in used_sources:
            continue
        for target in range(nt):
            if target in used_targets or not bool(candidate_mask[source, target]):
                continue
            if mode == "one_way" and not (
                int(source_prediction[source]) == target
                or int(target_prediction[target]) == source
            ):
                continue
            pair_logit = float(augmented_logits[source, target])
            margin = min(
                pair_logit - float(augmented_logits[source, nt]),
                pair_logit - float(augmented_logits[ns, target]),
            )
            if margin < margin_threshold:
                continue
            candidates.append(Link(source, target, pair_logit, margin, True))

    candidates.sort(
        key=lambda link: (
            -link.margin,
            -link.pair_logit,
            link.source_index,
            link.target_index,
        )
    )
    for link in candidates:
        if link.source_index in used_sources or link.target_index in used_targets:
            continue
        accepted.append(link)
        used_sources.add(link.source_index)
        used_targets.add(link.target_index)
    return accepted


def copy_without_tracks(source: h5py.File, target: h5py.File) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name in source.keys():
        if name not in {"row_track_ids", "accepted_links"}:
            source.copy(name, target)


def recover_track_file(
    source_path: Path,
    output_path: Path,
    mode: str,
    margin_threshold: float,
    force: bool,
    association_relation_path: Path | None = None,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    target_groups = (
        load_association_targets(association_relation_path)
        if association_relation_path is not None
        else None
    )
    if output_path.exists() and not force:
        raise FileExistsError(f"Recovered track file already exists: {output_path}")

    with h5py.File(source_path, "r", swmr=True) as source:
        if str(source.attrs.get("schema_version", "")) != "IBP-K360-predicted-tracklets-v1.1.0":
            raise ValueError(f"Unexpected predicted-track schema: {source_path}")
        original_track_ids = np.asarray(source["row_track_ids"], dtype=np.int64)
        sequence = str(source.attrs.get("sequence", ""))
        checkpoint_sha256 = str(source.attrs.get("source_checkpoint_sha256", ""))
        eligible_rows = np.flatnonzero(original_track_ids >= 0).astype(np.int64)
        disjoint = DisjointSet(len(original_track_ids))
        accepted_source: list[int] = []
        accepted_target: list[int] = []
        accepted_confidence: list[float] = []
        accepted_margin: list[float] = []
        accepted_fallback: list[int] = []
        original_links = fallback_links = 0
        association_correct = association_decisions = true_forward_links = 0
        correct_original_links = correct_fallback_links = 0
        groups = source["association_groups"]

        if target_groups is not None and len(target_groups) != len(groups["shape"]):
            raise ValueError("Track and relation files contain different association groups")

        for index, shape in enumerate(groups["shape"]):
            ns, nt = (int(value) for value in shape)
            source_rows = np.asarray(groups["source_rows"][index], dtype=np.int64)
            target_rows = np.asarray(groups["target_rows"][index], dtype=np.int64)
            logits = np.asarray(
                groups["raw_augmented_logits"][index], dtype=np.float32
            ).reshape(ns + 1, nt + 1)
            mask = np.asarray(
                groups["candidate_mask"][index], dtype=np.uint8
            ).reshape(ns, nt).astype(bool)
            links = select_links(logits, mask, mode, margin_threshold)
            source_targets = target_targets = None
            if target_groups is not None:
                target_group = target_groups[index]
                if target_group["shape"] != (ns, nt):
                    raise ValueError("Track and relation association shapes differ")
                if not np.array_equal(target_group["source_rows"], source_rows):
                    raise ValueError("Track and relation source rows differ")
                if not np.array_equal(target_group["target_rows"], target_rows):
                    raise ValueError("Track and relation target rows differ")
                source_targets = np.asarray(target_group["source_targets"], dtype=np.int64)
                target_targets = np.asarray(target_group["target_targets"], dtype=np.int64)
                source_prediction = logits[:ns].argmax(axis=1)
                target_prediction = logits[:, :nt].argmax(axis=0)
                association_correct += int((source_prediction == source_targets).sum())
                association_correct += int((target_prediction == target_targets).sum())
                association_decisions += ns + nt
                true_forward_links += int((source_targets < nt).sum())

            for link in links:
                source_row = int(source_rows[link.source_index])
                target_row = int(target_rows[link.target_index])
                disjoint.union(source_row, target_row)
                accepted_source.append(source_row)
                accepted_target.append(target_row)
                accepted_confidence.append(sigmoid(link.pair_logit))
                accepted_margin.append(link.margin)
                accepted_fallback.append(int(link.fallback))
                fallback_links += int(link.fallback)
                original_links += int(not link.fallback)
                if source_targets is not None and int(source_targets[link.source_index]) == link.target_index:
                    if link.fallback:
                        correct_fallback_links += 1
                    else:
                        correct_original_links += 1

        recovered_track_ids = compact_track_ids(disjoint, eligible_rows)
        members: dict[int, list[int]] = {}
        for row in eligible_rows:
            members.setdefault(int(recovered_track_ids[row]), []).append(int(row))
        row_frames = np.asarray(source["row_sample_indices"], dtype=np.int64)
        row_scenes = source["row_scene_tokens"].asstr()[:]
        for track_rows in members.values():
            frame_keys = {
                (str(row_scenes[row]), int(row_frames[row])) for row in track_rows
            }
            if len(frame_keys) != len(track_rows):
                raise ValueError("Recovered tracks contain two objects in one sample")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        with h5py.File(temporary, "w") as target:
            copy_without_tracks(source, target)
            target.attrs["recovery_schema_version"] = RECOVERY_SCHEMA_VERSION
            target.attrs["recovery_mode"] = mode
            target.attrs["recovery_margin_threshold"] = margin_threshold
            target.attrs["base_track_sha256"] = sha256_file(source_path)
            target.create_dataset(
                "row_track_ids", data=recovered_track_ids, compression="gzip"
            )
            links = target.create_group("accepted_links")
            links.create_dataset(
                "source_rows", data=np.asarray(accepted_source, dtype=np.int64)
            )
            links.create_dataset(
                "target_rows", data=np.asarray(accepted_target, dtype=np.int64)
            )
            links.create_dataset(
                "confidence", data=np.asarray(accepted_confidence, dtype=np.float32)
            )
            links.create_dataset(
                "dustbin_margin", data=np.asarray(accepted_margin, dtype=np.float32)
            )
            links.create_dataset(
                "recovery_fallback", data=np.asarray(accepted_fallback, dtype=np.uint8)
            )
        temporary.replace(output_path)

    lengths = [len(rows) for rows in members.values()]
    report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "sequence": sequence,
        "mode": mode,
        "margin_threshold": margin_threshold,
        "identity_labels_used_for_recovery": False,
        "ground_truth_association_targets_used_for_recovery": False,
        "base_track_file": str(source_path),
        "base_track_sha256": sha256_file(source_path),
        "output": str(output_path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "original_mutual_links": original_links,
        "fallback_links": fallback_links,
        "total_links": len(accepted_source),
        "predicted_tracks": len(members),
        "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "maximum_track_length": max(lengths, default=0),
    }
    if target_groups is not None:
        predicted_links = original_links + fallback_links
        correct_links = correct_original_links + correct_fallback_links
        report.update(
            {
                "association_relation_file": str(association_relation_path.resolve()),
                "association_correct": association_correct,
                "association_decisions": association_decisions,
                "association_accuracy": association_correct / max(association_decisions, 1),
                "true_forward_links": true_forward_links,
                "predicted_mutual_links": original_links,
                "correct_mutual_links": correct_original_links,
                "mutual_link_precision": correct_original_links / max(original_links, 1),
                "mutual_link_recall": correct_original_links / max(true_forward_links, 1),
                "predicted_recovery_links": predicted_links,
                "correct_recovery_links": correct_links,
                "correct_fallback_links": correct_fallback_links,
                "recovery_link_precision": correct_links / max(predicted_links, 1),
                "recovery_link_recall": correct_links / max(true_forward_links, 1),
            }
        )
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def threshold_name(value: float) -> str:
    return f"{value:.2f}".replace("-", "neg").replace(".", "p")


def build_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "recovery_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Recovery manifest already exists: {manifest_path}")
    variants: list[dict[str, Any]] = []
    for mode in args.modes:
        for margin in args.margin_thresholds:
            name = f"{mode}_margin_{threshold_name(float(margin))}"
            output_path = output_root / name / f"{args.source_track.name}"
            report = recover_track_file(
                args.source_track,
                output_path,
                mode,
                float(margin),
                args.force,
            )
            variants.append({"name": name, **report})
    manifest = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "selection_split": "validation",
        "held_out_test_opened": False,
        "source_track": str(args.source_track.resolve()),
        "source_track_sha256": sha256_file(args.source_track.resolve()),
        "variants": variants,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover fragmented validation tracklets from saved association logits."
    )
    parser.add_argument("--source-track", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=("one_way", "all"))
    parser.add_argument(
        "--margin-thresholds", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_sweep(parse_args()), indent=2))


if __name__ == "__main__":
    main()
