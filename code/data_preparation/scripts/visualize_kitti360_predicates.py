#!/usr/bin/env python3
"""Create deterministic visual and statistical audits for predicate V2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PREDICATES = [
    "left_of",
    "in_front_of",
    "near",
    "overlapping",
    "occluding",
]


def audit_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predicate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--examples-per-predicate", type=int, default=1)
    return parser.parse_args()


def bbox(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(np.isfinite(float(item)) for item in value):
        return None
    return [float(item) for item in value]


def centre(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return np.asarray(value, dtype=np.float64)


def relation_quality(frame: pd.DataFrame, predicate: str) -> pd.Series:
    if predicate == "left_of":
        return frame["delta_ego_xyz"].map(lambda value: abs(float(value[1])))
    if predicate == "in_front_of":
        return frame["delta_ego_xyz"].map(lambda value: abs(float(value[0])))
    if predicate == "near":
        return -frame["oriented_box_distance_m"].astype(float)
    if predicate == "overlapping":
        return frame["bev_iou"].astype(float)
    if predicate == "occluding":
        return frame["image_overlap_over_smaller"].astype(float)
    return pd.Series(0.0, index=frame.index)


def select_examples(
    relations: pd.DataFrame,
    annotations: pd.DataFrame,
    per_predicate: int,
) -> dict[str, list[pd.Series]]:
    visible_tokens = set(
        annotations.loc[
            annotations["projected_bbox_xyxy"].map(bbox).notna(), "token"
        ].astype(str)
    )
    selected: dict[str, list[pd.Series]] = {}
    for predicate in PREDICATES:
        candidates = relations[
            relations[predicate].astype(bool)
            & relations["supervised_eligible"].astype(bool)
            & relations["subject_annotation_token"].astype(str).isin(visible_tokens)
            & relations["object_annotation_token"].astype(str).isin(visible_tokens)
        ].copy()
        candidates["_quality"] = relation_quality(candidates, predicate)
        candidates = candidates.sort_values(
            ["_quality", "token"], ascending=[False, True]
        ).head(per_predicate)
        selected[predicate] = [row for _, row in candidates.iterrows()]
    return selected


def draw_relation_panel(
    dataset_root: Path,
    annotations: pd.DataFrame,
    row: pd.Series,
    predicate: str,
) -> Image.Image:
    annotation_index = annotations.set_index("token")
    subject = annotation_index.loc[row["subject_annotation_token"]]
    target = annotation_index.loc[row["object_annotation_token"]]
    image = Image.open(dataset_root / subject["cam0_filename"]).convert("RGB")
    image.thumbnail((640, 340), Image.Resampling.LANCZOS)

    original = Image.open(dataset_root / subject["cam0_filename"])
    scale_x = image.width / original.width
    scale_y = image.height / original.height
    original.close()
    draw = ImageDraw.Draw(image)
    font = audit_font(15)

    def scaled_box(value: object) -> tuple[int, int, int, int]:
        box = bbox(value)
        assert box is not None
        return (
            round(box[0] * scale_x),
            round(box[1] * scale_y),
            round(box[2] * scale_x),
            round(box[3] * scale_y),
        )

    subject_box = scaled_box(subject["projected_bbox_xyxy"])
    target_box = scaled_box(target["projected_bbox_xyxy"])
    draw.rectangle(subject_box, outline=(239, 68, 68), width=4)
    draw.rectangle(target_box, outline=(35, 170, 225), width=4)
    title = (
        f"{predicate}: {subject['raw_label']} -> {target['raw_label']}  "
        f"frame {int(subject['raw_frame_index'])}"
    )
    all_labels = ", ".join(str(value) for value in row["positive_predicates"])
    legend = (
        f"RED subject: {subject['raw_label']} | BLUE object: {target['raw_label']} "
        f"| all labels: {all_labels}"
    )
    title_box = draw.textbbox((0, 0), title, font=font)
    legend_box = draw.textbbox((0, 0), legend, font=font)
    line_height = max(title_box[3] - title_box[1], legend_box[3] - legend_box[1])
    draw.rectangle((0, 0, image.width, 2 * line_height + 14), fill=(255, 255, 255))
    draw.text((6, 3), title, fill=(0, 0, 0), font=font)
    draw.text((6, line_height + 8), legend, fill=(0, 0, 0), font=font)
    return image


def contact_sheet(
    dataset_root: Path,
    annotations: pd.DataFrame,
    selected: dict[str, list[pd.Series]],
    output: Path,
) -> dict[str, int]:
    panels: list[tuple[str, Image.Image | None]] = []
    support: dict[str, int] = {}
    for predicate in PREDICATES:
        examples = selected[predicate]
        support[predicate] = len(examples)
        if examples:
            for row in examples:
                panels.append(
                    (predicate, draw_relation_panel(dataset_root, annotations, row, predicate))
                )
        else:
            panels.append((predicate, None))

    columns = 4
    panel_width, panel_height = 640, 220
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = audit_font(16)
    for index, (predicate, panel) in enumerate(panels):
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        if panel is None:
            draw.rectangle((x, y, x + panel_width - 1, y + panel_height - 1), outline="gray")
            draw.text((x + 20, y + 30), f"{predicate}: no visible example", fill="black", font=font)
        else:
            sheet.paste(panel, (x, y + 25))
            draw.text((x + 5, y + 5), predicate, fill="black", font=font)
    sheet.save(output)
    return support


def plot_support(relations: pd.DataFrame, output: Path) -> dict[str, int]:
    counts = {name: int(relations[name].astype(bool).sum()) for name in PREDICATES}
    figure, axis = plt.subplots(figsize=(13, 6))
    values = [counts[name] for name in PREDICATES]
    bars = axis.bar(PREDICATES, values, color="#3579b9")
    axis.set_yscale("log")
    axis.set_ylabel("Positive ordered pairs (log scale)")
    axis.set_title("Canonical V3 spatial predicate support")
    axis.tick_params(axis="x", rotation=45)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return counts


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    predicate_root = args.predicate_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_json(predicate_root / "sample_annotations.json")
    relations = pd.read_json(predicate_root / "spatial_relations.json")
    selected = select_examples(relations, annotations, args.examples_per_predicate)
    visible_support = contact_sheet(
        dataset_root,
        annotations,
        selected,
        output_root / "predicate_contact_sheet.png",
    )
    counts = plot_support(relations, output_root / "predicate_support.png")

    report = {
        "candidate_only": True,
        "predicate_root": str(predicate_root),
        "annotation_count": int(len(annotations)),
        "relation_candidate_count": int(len(relations)),
        "predicate_positive_counts": counts,
        "visible_examples_selected": visible_support,
        "rare_predicates_below_100": [name for name, count in counts.items() if count < 100],
        "required_human_action": (
            "Inspect the contact sheet and freeze thresholds using training sequences only."
        ),
    }
    (output_root / "predicate_visual_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved visual audit to: {output_root}")


if __name__ == "__main__":
    main()
