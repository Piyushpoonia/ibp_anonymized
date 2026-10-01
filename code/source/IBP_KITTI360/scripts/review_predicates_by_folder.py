#!/usr/bin/env python3
"""Create predicate-named KITTI-360 human-review folders and correction JSON."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd

from review_random_predicates import (
    SPATIAL_PREDICATES,
    TEMPORAL_PREDICATES,
    draw_spatial_review,
    iter_json_array,
    reservoir_groups,
    temporal_contact_sheet,
)


def temporal_reservoirs(
    path: Path,
    group_sizes: dict[str, int],
    seed: int,
    minimum_joint_visible_steps: int,
) -> dict[str, list[dict[str, object]]]:
    reservoirs = {name: [] for name in group_sizes}
    seen = {name: 0 for name in group_sizes}
    generators = {
        name: random.Random(seed + 2029 * index)
        for index, name in enumerate(group_sizes)
    }
    for row in iter_json_array(path):
        if not bool(row.get("supervised_eligible", False)):
            continue
        joint_visible_steps = int(row.get("joint_visible_steps", 0))
        if not bool(row.get("human_review_eligible", False)):
            continue
        if joint_visible_steps < minimum_joint_visible_steps:
            continue
        for name, requested in group_sizes.items():
            if requested <= 0 or not bool(row.get(name, False)):
                continue
            seen[name] += 1
            reservoir = reservoirs[name]
            if len(reservoir) < requested:
                reservoir.append(row)
                continue
            replacement = generators[name].randrange(seen[name])
            if replacement < requested:
                reservoir[replacement] = row
    for records in reservoirs.values():
        records.sort(key=lambda row: str(row["token"]))
    return reservoirs


def correction_record(
    row: dict[str, object],
    kind: str,
    sampled_for: str,
    image_name: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "relation_id": str(row["token"]),
        "sequence": str(row["sequence"]),
        "kind": kind,
        "sampled_for_predicate": sampled_for,
        "subject_label": str(row["subject_label"]),
        "object_label": str(row["object_label"]),
        "automatic_predicates": list(row.get("positive_predicates", [])),
        "image_filename": image_name,
        "human_decision": None,
        "corrected_predicates": [],
        "human_notes": "",
    }
    if kind == "spatial":
        record["raw_frame_index"] = int(row["raw_frame_index"])
    else:
        record["raw_frame_indices"] = [int(value) for value in row["raw_frame_indices"]]
        radial_velocity = row.get("subject_radial_velocity_toward_object_mps")
        record["subject_radial_velocity_toward_object_mps"] = (
            float(radial_velocity) if radial_velocity is not None else None
        )
        record["distance_change_m"] = float(row["distance_change_m"])
        record["decreasing_step_fraction"] = float(row["decreasing_step_fraction"])
        record["increasing_step_fraction"] = float(row["increasing_step_fraction"])
        record["subject_visible_steps"] = int(row["subject_visible_steps"])
        record["object_visible_steps"] = int(row["object_visible_steps"])
        record["joint_visible_steps"] = int(row["joint_visible_steps"])
        record["subject_motion_toward_object"] = bool(
            row["subject_motion_toward_object"]
        )
        record["subject_motion_away_from_object"] = bool(
            row["subject_motion_away_from_object"]
        )
        record["pair_distance_decreasing"] = bool(
            row["pair_distance_decreasing"]
        )
        record["pair_distance_increasing"] = bool(row["pair_distance_increasing"])
    return record


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def render_group(
    dataset_root: Path,
    output_root: Path,
    kind: str,
    predicate: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    folder_name = predicate if predicate != "no_relation" else f"{kind}_no_relation"
    folder = output_root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    corrections: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        relation_id = str(row["token"])
        image_name = f"{index:03d}_relation_{relation_id}.png"
        image_path = folder / image_name
        if kind == "spatial":
            draw_spatial_review(dataset_root, row, predicate).save(image_path)
        else:
            temporal_contact_sheet(
                dataset_root,
                pd.Series(row),
                predicate,
                image_path,
            )
        corrections.append(
            correction_record(row, kind, predicate, image_name)
        )
    write_json(folder / "corrections.json", corrections)
    return {
        "folder": folder_name,
        "kind": kind,
        "predicate": predicate,
        "examples_written": len(corrections),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predicate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--examples-per-predicate", type=int, default=50)
    parser.add_argument("--no-relation-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assigned-sequence", required=True)
    parser.add_argument("--minimum-joint-visible-steps", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    predicate_root = args.predicate_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.name != args.assigned_sequence:
        raise ValueError("Output folder name must equal --assigned-sequence")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(f"Review output already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    spatial_sizes = {name: args.examples_per_predicate for name in SPATIAL_PREDICATES}
    spatial_sizes["no_relation"] = args.no_relation_examples
    temporal_sizes = {name: args.examples_per_predicate for name in TEMPORAL_PREDICATES}
    temporal_sizes["no_relation"] = args.no_relation_examples

    spatial = reservoir_groups(
        predicate_root / "spatial_relations.json",
        spatial_sizes,
        args.seed,
    )
    temporal = temporal_reservoirs(
        predicate_root / "temporal" / "temporal_relations.json",
        temporal_sizes,
        args.seed + 100_003,
        args.minimum_joint_visible_steps,
    )

    groups: list[dict[str, object]] = []
    for predicate in [*SPATIAL_PREDICATES, "no_relation"]:
        groups.append(
            render_group(
                dataset_root,
                output_root,
                "spatial",
                predicate,
                spatial[predicate],
            )
        )
    for predicate in [*TEMPORAL_PREDICATES, "no_relation"]:
        groups.append(
            render_group(
                dataset_root,
                output_root,
                "temporal",
                predicate,
                temporal[predicate],
            )
        )

    manifest = {
        "candidate_only": True,
        "sequence": args.assigned_sequence,
        "seed": args.seed,
        "examples_per_predicate_requested": args.examples_per_predicate,
        "no_relation_examples_requested": args.no_relation_examples,
        "spatial_predicates": SPATIAL_PREDICATES,
        "temporal_predicates": TEMPORAL_PREDICATES,
        "temporal_review_filter": {
            "requires_human_review_eligible": True,
            "minimum_joint_visible_steps": args.minimum_joint_visible_steps,
        },
        "total_examples_written": sum(int(group["examples_written"]) for group in groups),
        "groups": groups,
        "correction_instructions": {
            "human_decision_values": ["correct", "incorrect", "ambiguous"],
            "incorrect": (
                "Set corrected_predicates to the complete corrected list. Use [] for no relation."
            ),
            "relation_id": "The relation ID appears in both the PNG and corrections.json.",
        },
    }
    write_json(output_root / "review_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"Saved predicate-folder review to: {output_root}")


if __name__ == "__main__":
    main()
