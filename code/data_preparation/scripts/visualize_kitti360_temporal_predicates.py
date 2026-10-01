#!/usr/bin/env python3
"""Visualize candidate five-observation temporal relations in world BEV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICATES = [
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
]


def select_example(relations: pd.DataFrame, predicate: str) -> pd.Series | None:
    candidates = relations[
        relations[predicate].astype(bool)
        & relations["supervised_eligible"].astype(bool)
    ].copy()
    if candidates.empty:
        return None
    if predicate == "approaching":
        return candidates.sort_values(["distance_change_m", "token"]).iloc[0]
    if predicate == "moving_away":
        return candidates.sort_values(
            ["distance_change_m", "token"], ascending=[False, True]
        ).iloc[0]
    if predicate == "same_motion_direction":
        return candidates.sort_values(
            ["motion_direction_cosine", "token"], ascending=[False, True]
        ).iloc[0]
    return candidates.sort_values(["observed_path_distance_m", "token"]).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    temporal_root = args.temporal_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tracklets = pd.read_json(temporal_root / "tracklet_windows.json")
    relations = pd.read_json(temporal_root / "temporal_relations.json")
    tracklet_index = tracklets.set_index("token")
    counts = {name: int(relations[name].astype(bool).sum()) for name in PREDICATES}

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    examples: dict[str, str | None] = {}
    for axis, predicate in zip(axes.flat, PREDICATES):
        relation = select_example(relations, predicate)
        axis.set_title(f"{predicate} (support={counts[predicate]})")
        axis.set_xlabel("world x offset (m)")
        axis.set_ylabel("world y offset (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        if relation is None:
            axis.text(
                0.5,
                0.5,
                "No genuine example in this drive",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            examples[predicate] = None
            continue
        subject = tracklet_index.loc[relation["subject_tracklet_token"]]
        target = tracklet_index.loc[relation["object_tracklet_token"]]
        path_a = np.asarray(subject["centers_world"], dtype=float).reshape(5, 3)[:, :2]
        path_b = np.asarray(target["centers_world"], dtype=float).reshape(5, 3)[:, :2]
        origin = np.vstack([path_a, path_b]).mean(axis=0)
        path_a = path_a - origin
        path_b = path_b - origin
        axis.plot(path_a[:, 0], path_a[:, 1], "o-", color="#d94841", label=f"subject: {subject['raw_label']}")
        axis.plot(path_b[:, 0], path_b[:, 1], "o-", color="#2479b9", label=f"object: {target['raw_label']}")
        axis.scatter(path_a[0, 0], path_a[0, 1], marker="s", s=90, color="#d94841")
        axis.scatter(path_b[0, 0], path_b[0, 1], marker="s", s=90, color="#2479b9")
        axis.scatter(path_a[-1, 0], path_a[-1, 1], marker="*", s=150, color="#d94841")
        axis.scatter(path_b[-1, 0], path_b[-1, 1], marker="*", s=150, color="#2479b9")
        axis.legend(loc="best")
        axis.text(
            0.02,
            0.02,
            f"distance change={relation['distance_change_m']:.2f} m\n"
            f"direction cosine={relation['motion_direction_cosine']:.2f}",
            transform=axis.transAxes,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        examples[predicate] = str(relation["token"])
    figure.suptitle("Five-observation KITTI-360 temporal-predicate audit")
    figure.tight_layout()
    figure.savefig(output_root / "temporal_predicate_examples.png", dpi=180)
    plt.close(figure)

    report = {
        "candidate_only": True,
        "tracklet_windows": int(len(tracklets)),
        "temporal_relation_candidates": int(len(relations)),
        "no_relation_candidates": int(relations["no_relation"].sum()),
        "predicate_positive_counts": counts,
        "selected_example_tokens": examples,
        "unsupported_in_this_partition": [name for name, count in counts.items() if count == 0],
    }
    (output_root / "temporal_visual_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved temporal audit to: {output_root}")


if __name__ == "__main__":
    main()
