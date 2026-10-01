#!/usr/bin/env python3
"""Build hidden adjacent-sample association targets for KITTI-360 objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "IBP-K360-association-v2.0.0-candidate"


def stable_token(*parts: object) -> str:
    source = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(source, digest_size=16).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def annotation_links(
    annotations: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_meta = samples[["token", "scene_token", "sample_index"]].rename(
        columns={"token": "sample_token", "scene_token": "timeline_scene_token"}
    )
    merged = annotations.merge(sample_meta, on="sample_token", how="inner")
    if len(merged) != len(annotations):
        raise ValueError("Some annotations do not resolve to timeline samples")
    if not (merged["scene_token"] == merged["timeline_scene_token"]).all():
        raise ValueError("Annotation and timeline scene tokens disagree")

    link_rows: list[dict[str, Any]] = []
    for (scene_token, instance_token), group in merged.groupby(
        ["scene_token", "instance_token"], sort=True
    ):
        ordered = group.sort_values("sample_index")
        records = list(ordered.itertuples(index=False))
        for position, record in enumerate(records):
            previous = records[position - 1] if position else None
            following = records[position + 1] if position + 1 < len(records) else None
            previous_token = (
                str(previous.token)
                if previous is not None
                and int(record.sample_index) - int(previous.sample_index) == 1
                else ""
            )
            next_token = (
                str(following.token)
                if following is not None
                and int(following.sample_index) - int(record.sample_index) == 1
                else ""
            )
            link_rows.append(
                {
                    "annotation_token": str(record.token),
                    "sample_token": str(record.sample_token),
                    "scene_token": str(scene_token),
                    "sample_index": int(record.sample_index),
                    "prev": previous_token,
                    "next": next_token,
                    "has_prev": bool(previous_token),
                    "has_next": bool(next_token),
                    "schema_version": SCHEMA_VERSION,
                }
            )
    links = pd.DataFrame(link_rows)
    if not links["annotation_token"].is_unique:
        raise ValueError("Annotation-link tokens are not unique")
    return merged, links


def pair_row(
    source: Any,
    target: Any,
    source_sample: str,
    target_sample: str,
    distance: float,
    is_match: bool,
    forced_positive: bool,
) -> dict[str, Any]:
    source_center = np.asarray(source.center_world, dtype=np.float64)
    target_center = np.asarray(target.center_world, dtype=np.float64)
    return {
        "token": stable_token("association-pair-v2", source.token, target.token),
        "source_sample_token": source_sample,
        "target_sample_token": target_sample,
        "source_annotation_token": str(source.token),
        "target_annotation_token": str(target.token),
        "candidate_type": "object_pair",
        "source_label": str(source.raw_label),
        "target_label": str(target.raw_label),
        "relative_world_xyz": (target_center - source_center).tolist(),
        "center_distance_m": float(distance),
        "is_match": bool(is_match),
        "is_dustbin": False,
        "forced_positive_outside_gate": bool(forced_positive),
        "forward_target": bool(is_match),
        "reverse_target": bool(is_match),
        "supervised_eligible": bool(
            source.supervised_eligible and target.supervised_eligible
        ),
        "schema_version": SCHEMA_VERSION,
    }


def dustbin_row(
    source: Any | None,
    target: Any | None,
    source_sample: str,
    target_sample: str,
    positive: bool,
) -> dict[str, Any]:
    if source is not None:
        token = stable_token("association-source-dustbin-v2", source.token, target_sample)
        candidate_type = "source_to_dustbin"
        source_token, target_token = str(source.token), ""
        source_label, target_label = str(source.raw_label), "__dustbin__"
        forward_target, reverse_target = bool(positive), False
        supervised_eligible = bool(source.supervised_eligible)
    else:
        assert target is not None
        token = stable_token("association-dustbin-target-v2", source_sample, target.token)
        candidate_type = "dustbin_to_target"
        source_token, target_token = "", str(target.token)
        source_label, target_label = "__dustbin__", str(target.raw_label)
        forward_target, reverse_target = False, bool(positive)
        supervised_eligible = bool(target.supervised_eligible)
    return {
        "token": token,
        "source_sample_token": source_sample,
        "target_sample_token": target_sample,
        "source_annotation_token": source_token,
        "target_annotation_token": target_token,
        "candidate_type": candidate_type,
        "source_label": source_label,
        "target_label": target_label,
        "relative_world_xyz": [0.0, 0.0, 0.0],
        "center_distance_m": float("nan"),
        "is_match": False,
        "is_dustbin": True,
        "forced_positive_outside_gate": False,
        "forward_target": forward_target,
        "reverse_target": reverse_target,
        "supervised_eligible": supervised_eligible,
        "schema_version": SCHEMA_VERSION,
    }


def build_candidates(
    merged: pd.DataFrame,
    samples: pd.DataFrame,
    gate_distance: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    annotations_by_sample = {
        str(sample_token): list(group.itertuples(index=False))
        for sample_token, group in merged.groupby("sample_token", sort=False)
    }
    rows: list[dict[str, Any]] = []
    adjacent_pairs = 0
    forced_positives = 0
    for sample in samples.sort_values(["scene_token", "sample_index"]).itertuples(index=False):
        source_sample = str(sample.token)
        target_sample = str(sample.next)
        if not target_sample:
            continue
        sources = annotations_by_sample.get(source_sample, [])
        targets = annotations_by_sample.get(target_sample, [])
        if not sources and not targets:
            continue
        adjacent_pairs += 1
        source_identities = {str(item.instance_token): item for item in sources}
        target_identities = {str(item.instance_token): item for item in targets}

        for source in sources:
            source_center = np.asarray(source.center_world, dtype=np.float64)
            matching_target = target_identities.get(str(source.instance_token))
            included_match = False
            for target in targets:
                target_center = np.asarray(target.center_world, dtype=np.float64)
                distance = float(np.linalg.norm((target_center - source_center)[:2]))
                is_match = str(source.instance_token) == str(target.instance_token)
                if distance > gate_distance and not is_match:
                    continue
                forced = bool(is_match and distance > gate_distance)
                forced_positives += int(forced)
                included_match = included_match or is_match
                rows.append(
                    pair_row(
                        source,
                        target,
                        source_sample,
                        target_sample,
                        distance,
                        is_match,
                        forced,
                    )
                )
            if matching_target is not None and not included_match:
                raise AssertionError("A true match was lost from candidate generation")
            rows.append(
                dustbin_row(
                    source,
                    None,
                    source_sample,
                    target_sample,
                    matching_target is None,
                )
            )

        for target in targets:
            rows.append(
                dustbin_row(
                    None,
                    target,
                    source_sample,
                    target_sample,
                    str(target.instance_token) not in source_identities,
                )
            )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No adjacent-frame association candidates were generated")
    if not candidates["token"].is_unique:
        raise ValueError("Association candidate tokens are not unique")

    object_pairs = candidates[candidates["candidate_type"] == "object_pair"]
    source_dustbin = candidates[candidates["candidate_type"] == "source_to_dustbin"]
    target_dustbin = candidates[candidates["candidate_type"] == "dustbin_to_target"]
    counts = {
        "adjacent_sample_pairs": adjacent_pairs,
        "association_candidates": int(len(candidates)),
        "object_pair_candidates": int(len(object_pairs)),
        "positive_identity_pairs": int(object_pairs["is_match"].sum()),
        "negative_identity_pairs": int((~object_pairs["is_match"]).sum()),
        "source_dustbin_candidates": int(len(source_dustbin)),
        "positive_disappearances": int(source_dustbin["forward_target"].sum()),
        "target_dustbin_candidates": int(len(target_dustbin)),
        "positive_births": int(target_dustbin["reverse_target"].sum()),
        "forced_positive_outside_gate": forced_positives,
    }
    return candidates, counts


def validate_targets(candidates: pd.DataFrame) -> dict[str, Any]:
    forward = candidates[
        candidates["candidate_type"].isin(["object_pair", "source_to_dustbin"])
    ]
    reverse = candidates[
        candidates["candidate_type"].isin(["object_pair", "dustbin_to_target"])
    ]
    forward_counts = forward.groupby(
        ["source_sample_token", "target_sample_token", "source_annotation_token"]
    )["forward_target"].sum()
    reverse_counts = reverse.groupby(
        ["source_sample_token", "target_sample_token", "target_annotation_token"]
    )["reverse_target"].sum()
    bad_forward = int((forward_counts != 1).sum())
    bad_reverse = int((reverse_counts != 1).sum())
    return {
        "passed": bad_forward == 0 and bad_reverse == 0,
        "forward_groups_with_nonunique_target": bad_forward,
        "reverse_groups_with_nonunique_target": bad_reverse,
        "unique_candidate_tokens": bool(candidates["token"].is_unique),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--predicate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-distance", type=float, default=15.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    timeline_root = args.timeline_root.resolve()
    predicate_root = args.predicate_root.resolve()
    output_root = args.output_root.resolve()
    result_file = output_root / "association_build_report.json"
    if result_file.exists() and not args.force:
        raise FileExistsError(f"Output exists: {result_file}. Pass --force to replace.")
    output_root.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(timeline_root / "tables/sample.parquet")
    annotations = pd.read_json(predicate_root / "sample_annotations.json")
    merged, links = annotation_links(annotations, samples)
    candidates, counts = build_candidates(merged, samples, args.gate_distance)
    validation = validate_targets(candidates)
    if not validation["passed"]:
        raise AssertionError(f"Association target validation failed: {validation}")

    atomic_parquet(links, output_root / "annotation_temporal_link.parquet")
    atomic_parquet(candidates, output_root / "association_candidate.parquet")
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_distance_m": args.gate_distance,
        "counts": counts,
        "validation": validation,
        "privacy_contract": (
            "instance_token is used only to construct is_match and dustbin targets; "
            "it must not be exposed to the association model input."
        ),
    }
    atomic_json(result_file, report)
    print(json.dumps(report, indent=2))
    print(f"Saved association targets to: {output_root}")


if __name__ == "__main__":
    main()
