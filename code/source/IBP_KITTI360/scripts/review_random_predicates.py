#!/usr/bin/env python3
"""Export a deterministic random predicate set for human verification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from visualize_kitti360_predicates import bbox


SPATIAL_PREDICATES = [
    "left_of",
    "in_front_of",
    "near",
    "overlapping",
    "occluding",
]
TEMPORAL_PREDICATES = [
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
]


def sample_rows(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.iloc[:0].copy()
    return frame.sample(n=min(count, len(frame)), random_state=seed).sort_values("token")


def iter_json_array(path: Path) -> Iterator[dict[str, object]]:
    """Stream arrays written by atomic_json_records, one record per line."""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped in {"", "[", "]"}:
                continue
            if stripped.endswith(","):
                stripped = stripped[:-1]
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object record in {path}")
            yield value


def reservoir_groups(
    path: Path,
    group_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    reservoirs = {name: [] for name in group_sizes}
    seen = {name: 0 for name in group_sizes}
    generators = {
        name: random.Random(seed + 1009 * index)
        for index, name in enumerate(group_sizes)
    }
    for row in iter_json_array(path):
        if not bool(row.get("supervised_eligible", False)):
            continue
        if bbox(row.get("subject_projected_bbox_xyxy")) is None:
            continue
        if bbox(row.get("object_projected_bbox_xyxy")) is None:
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


def spatial_record(row: pd.Series, predicate: str, image: Path) -> dict[str, object]:
    return {
        "relation_token": str(row["token"]),
        "sequence": str(row["sequence"]),
        "raw_frame_index": int(row["raw_frame_index"]),
        "reviewed_as": predicate,
        "subject_label": str(row["subject_label"]),
        "object_label": str(row["object_label"]),
        "positive_predicates": list(row["positive_predicates"]),
        "horizontal_center_distance_m": float(row["horizontal_center_distance_m"]),
        "oriented_box_distance_m": float(row["oriented_box_distance_m"]),
        "bev_iou": float(row["bev_iou"]),
        "image_overlap_over_smaller": float(row["image_overlap_over_smaller"]),
        "image": str(image).replace("\\", "/"),
        "human_valid": None,
        "human_corrected_predicates": [],
        "human_notes": "",
    }


def draw_spatial_review(
    dataset_root: Path,
    row: dict[str, object],
    predicate: str,
) -> Image.Image:
    source = dataset_root / Path(str(row["rgb_left_filename"]))
    image = Image.open(source).convert("RGB")
    draw_review_box(
        image,
        row.get("subject_projected_bbox_xyxy"),
        f"SUBJECT: {row['subject_label']}",
        "#d62728",
    )
    draw_review_box(
        image,
        row.get("object_projected_bbox_xyxy"),
        f"OBJECT: {row['object_label']}",
        "#1677b8",
    )
    image.thumbnail((960, 520), Image.Resampling.LANCZOS)
    header = Image.new("RGB", (image.width, 102), "white")
    draw = ImageDraw.Draw(header)
    draw.text(
        (8, 7),
        f"{predicate}: {row['subject_label']} -> {row['object_label']} | "
        f"frame {int(row['raw_frame_index'])}",
        fill="black",
    )
    all_labels = ", ".join(str(value) for value in row["positive_predicates"])
    draw.text((8, 31), f"all automatic labels: {all_labels or 'no_relation'}", fill="black")
    draw.text(
        (8, 54),
        "Canonical: right/behind/occluded_by are represented on the reverse ordered edge.",
        fill="black",
    )
    draw.text((8, 77), f"relation ID: {row['token']}", fill="black")
    panel = Image.new("RGB", (image.width, image.height + header.height), "white")
    panel.paste(header, (0, 0))
    panel.paste(image, (0, header.height))
    return panel


def export_spatial(
    dataset_root: Path,
    predicate_root: Path,
    output_root: Path,
    count: int,
    no_relation_count: int,
    seed: int,
) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    for old_image in output_root.glob("*.png"):
        old_image.unlink()
    review: list[dict[str, object]] = []
    group_sizes = {name: count for name in SPATIAL_PREDICATES}
    group_sizes["no_relation"] = no_relation_count
    selected = reservoir_groups(predicate_root / "spatial_relations.json", group_sizes, seed)
    for predicate, records in selected.items():
        for example_index, row in enumerate(records, start=1):
            filename = f"{predicate}_{example_index:03d}_{row['token']}.png"
            image_path = output_root / filename
            panel = draw_spatial_review(dataset_root, row, predicate)
            panel.save(image_path)
            review.append(spatial_record(row, predicate, image_path.relative_to(output_root.parent)))
    return review


def draw_review_box(
    image: Image.Image,
    value: object,
    label: str,
    color: str,
) -> bool:
    coordinates = bbox(value)
    if coordinates is None:
        return False
    draw = ImageDraw.Draw(image)
    xyxy = tuple(round(number) for number in coordinates)
    draw.rectangle(xyxy, outline=color, width=5)
    x, y = xyxy[:2]
    text = f"{label}"
    text_box = draw.textbbox((0, 0), text, font=ImageFont.load_default())
    draw.rectangle(
        (x, max(0, y - 17), x + text_box[2] + 8, max(0, y - 17) + 16),
        fill="white",
    )
    draw.text((x + 4, max(0, y - 15)), text, fill=color, font=ImageFont.load_default())
    return True


def temporal_contact_sheet(
    dataset_root: Path,
    relation: pd.Series,
    predicate: str,
    output: Path,
) -> None:
    required = (
        "raw_frame_indices",
        "rgb_left_filenames",
        "subject_projected_bboxes_xyxy",
        "object_projected_bboxes_xyxy",
        "relative_distances_m",
    )
    if any(len(relation[key]) != 5 for key in required):
        raise ValueError(f"Temporal relation {relation['token']} does not contain five frames")

    panel_width = 640
    header_height = 168
    panels: list[Image.Image] = []
    for index in range(5):
        source = dataset_root / Path(str(relation["rgb_left_filenames"][index]))
        image = Image.open(source).convert("RGB")
        subject_visible = draw_review_box(
            image,
            relation["subject_projected_bboxes_xyxy"][index],
            f"SUBJECT: {relation['subject_label']}",
            "#d62728",
        )
        object_visible = draw_review_box(
            image,
            relation["object_projected_bboxes_xyxy"][index],
            f"OBJECT: {relation['object_label']}",
            "#1677b8",
        )
        image_height = round(image.height * panel_width / image.width)
        image = image.resize((panel_width, image_height), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_width, image_height + 52), "white")
        panel.paste(image, (0, 52))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (7, 6),
            f"Step {index + 1}/5 | frame {int(relation['raw_frame_indices'][index])} | "
            f"distance {float(relation['relative_distances_m'][index]):.2f} m",
            fill="black",
        )
        draw.text(
            (7, 28),
            f"red subject visible: {subject_visible} | blue object visible: {object_visible}",
            fill="black",
        )
        panels.append(panel)

    tile_height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (panel_width * 2, header_height + tile_height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 8),
        f"Temporal: {relation['subject_label']} -> {relation['object_label']} | {predicate}",
        fill="black",
    )
    draw.text(
        (10, 30),
        f"distance change {float(relation['distance_change_m']):.2f} m | "
        f"speed {float(relation['subject_speed_mps']):.2f} / "
        f"{float(relation['object_speed_mps']):.2f} m/s",
        fill="black",
    )
    radial_velocity = relation.get("subject_radial_velocity_toward_object_mps")
    radial_text = (
        f"{float(radial_velocity):.2f} m/s"
        if radial_velocity is not None and pd.notna(radial_velocity)
        else "unavailable in legacy annotation"
    )
    draw.text(
        (10, 52),
        f"subject radial velocity toward object: {radial_text}",
        fill="black",
    )
    draw.text(
        (10, 74),
        f"trend: decreasing {float(relation['decreasing_step_fraction']):.2f} | "
        f"increasing {float(relation['increasing_step_fraction']):.2f}",
        fill="black",
    )
    draw.text(
        (10, 96),
        f"joint RGB visibility: {int(relation.get('joint_visible_steps', 0))}/5 | "
        f"review eligible: {bool(relation.get('human_review_eligible', False))}",
        fill="black",
    )
    draw.text(
        (10, 118),
        "Rule: approaching = subject toward + distance decreasing; "
        "moving_away = subject away + distance increasing",
        fill="black",
    )
    draw.text((10, 140), f"relation token: {relation['token']}", fill="black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * panel_width, header_height + (index // 2) * tile_height))
    sheet.save(output)


def temporal_record(row: pd.Series, predicate: str, image: Path) -> dict[str, object]:
    path_distance = row["observed_path_distance_m"]
    return {
        "relation_token": str(row["token"]),
        "sequence": str(row["sequence"]),
        "raw_frame_indices": [int(item) for item in row["raw_frame_indices"]],
        "middle_raw_frame_index": int(row["middle_raw_frame_index"]),
        "reviewed_as": predicate,
        "subject_label": str(row["subject_label"]),
        "object_label": str(row["object_label"]),
        "positive_predicates": list(row["positive_predicates"]),
        "relative_distances_m": [float(item) for item in row["relative_distances_m"]],
        "distance_change_m": float(row["distance_change_m"]),
        "decreasing_step_fraction": float(row["decreasing_step_fraction"]),
        "increasing_step_fraction": float(row["increasing_step_fraction"]),
        "joint_visible_steps": int(row.get("joint_visible_steps", 0)),
        "human_review_eligible": bool(row.get("human_review_eligible", False)),
        "subject_radial_velocity_toward_object_mps": float(
            row["subject_radial_velocity_toward_object_mps"]
        ),
        "motion_direction_cosine": float(row["motion_direction_cosine"]),
        "observed_path_distance_m": (
            float(path_distance) if pd.notna(path_distance) else None
        ),
        "image": str(image).replace("\\", "/"),
        "human_valid": None,
        "human_corrected_predicates": [],
        "human_notes": "",
    }


def export_temporal(
    dataset_root: Path,
    predicate_root: Path,
    output_root: Path,
    count: int,
    no_relation_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[str]]:
    temporal_root = predicate_root / "temporal"
    relations = pd.read_json(temporal_root / "temporal_relations.json")
    if "human_review_eligible" in relations:
        relations = relations[relations["human_review_eligible"].astype(bool)]
    output_root.mkdir(parents=True, exist_ok=True)
    for old_image in output_root.glob("*.png"):
        old_image.unlink()

    review: list[dict[str, object]] = []
    unsupported: list[str] = []
    groups = [(name, relations[relations[name].astype(bool)], count) for name in TEMPORAL_PREDICATES]
    groups.append(("no_relation", relations[relations["no_relation"].astype(bool)], no_relation_count))
    for group_index, (predicate, candidates, requested) in enumerate(groups):
        if candidates.empty and predicate != "no_relation":
            unsupported.append(predicate)
            continue
        selected = sample_rows(candidates, requested, seed + 2029 * group_index)
        for example_index, (_, row) in enumerate(selected.iterrows(), start=1):
            filename = f"{predicate}_{example_index:03d}_{row['token']}.png"
            image_path = output_root / filename
            temporal_contact_sheet(dataset_root, row, predicate, image_path)
            review.append(temporal_record(row, predicate, image_path.relative_to(output_root.parent)))
    return review, unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predicate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--examples-per-predicate", type=int, default=10)
    parser.add_argument("--no-relation-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reviewer-id", default="unassigned")
    parser.add_argument("--assigned-sequence", default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    predicate_root = args.predicate_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    spatial = export_spatial(
        dataset_root,
        predicate_root,
        output_root / "spatial_images",
        args.examples_per_predicate,
        args.no_relation_examples,
        args.seed,
    )
    temporal, unsupported = export_temporal(
        dataset_root,
        predicate_root,
        output_root / "temporal_images",
        args.examples_per_predicate,
        args.no_relation_examples,
        args.seed,
    )
    assigned_sequence = args.assigned_sequence or predicate_root.name
    for record in spatial + temporal:
        record["reviewer_id"] = args.reviewer_id
        record["assigned_sequence"] = assigned_sequence
    (output_root / "spatial_review.json").write_text(
        json.dumps(spatial, indent=2), encoding="utf-8"
    )
    (output_root / "temporal_review.json").write_text(
        json.dumps(temporal, indent=2), encoding="utf-8"
    )
    summary = {
        "reviewer_id": args.reviewer_id,
        "assigned_sequence": assigned_sequence,
        "seed": args.seed,
        "examples_per_predicate_requested": args.examples_per_predicate,
        "no_relation_examples_requested": args.no_relation_examples,
        "spatial_examples_written": len(spatial),
        "temporal_examples_written": len(temporal),
        "unsupported_temporal_predicates": unsupported,
        "spatial_predicates": SPATIAL_PREDICATES,
        "temporal_predicates": TEMPORAL_PREDICATES,
        "canonical_inverse_encoding": {
            "right_of(A,B)": "left_of(B,A)",
            "behind(A,B)": "in_front_of(B,A)",
            "occluded_by(A,B)": "occluding(B,A)",
        },
        "review_instruction": (
            "Inspect each image, then set human_valid, human_corrected_predicates, "
            "and human_notes in the corresponding review JSON record. Corrected labels "
            "must use only the five canonical spatial or four temporal names."
        ),
    }
    (output_root / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved random human-review package to: {output_root}")


if __name__ == "__main__":
    main()
