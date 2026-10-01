"""Build the frozen IBP-K360 timeline, labels, features, and relation shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "code" / "data_preparation"
SOURCE = ROOT / "code" / "source"
WEIGHTS = ROOT / "resource"
SPLIT = SOURCE / "IBP_KITTI360" / "splits" / "kitti360_ibp_sequence_split_v1.json"


def sequences() -> list[str]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return split["train"] + split["validation"] + split["test"]


def check_raw(root: Path, names: list[str]) -> None:
    required = [
        root / "calibration" / "perspective.txt",
        root / "calibration" / "calib_cam_to_pose.txt",
        root / "calibration" / "calib_cam_to_velo.txt",
    ]
    for name in names:
        required.extend(
            [
                root / "data_2d_raw" / name / "image_00" / "data_rect",
                root / "data_2d_semantics" / "train" / name / "image_00" / "semantic",
                root / "data_3d_raw" / name / "velodyne_points" / "data",
                root / "data_poses" / name,
                root / "data_3d_bboxes" / "train" / f"{name}.xml",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Incomplete KITTI-360 raw dataset; first missing paths:\n" + "\n".join(missing[:12]))


def check_weights() -> None:
    pointnet = WEIGHTS / "obj_enc.pth"
    if not pointnet.is_file():
        raise SystemExit(f"Missing PointNet checkpoint: {pointnet}")
    hub = WEIGHTS / "hf_home" / "hub"
    for name in ("models--openai--clip-vit-base-patch16", "models--Salesforce--blip-image-captioning-base"):
        if not (hub / name).is_dir():
            raise SystemExit(f"Missing frozen model cache: {hub / name}")


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "KITTI-360")
    parser.add_argument("--output-root", type=Path, default=ROOT / "PREPARED_DATASET")
    parser.add_argument("--step", choices=("all", "timeline", "predicates", "features", "relations"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Regenerate an existing output; this can be expensive.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw = args.dataset_root.resolve()
    output = args.output_root.resolve()
    names = sequences()
    if not args.dry_run:
        check_raw(raw, names)
        if args.step in ("all", "features"):
            subprocess.run([sys.executable, str(WEIGHTS / "install_weights.py")], check=True)
            check_weights()
        output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = str(WEIGHTS / "hf_home")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    py = sys.executable
    for name in names:
        timeline = output / "timeline" / "parts" / name
        predicates = output / "predicates" / "parts" / name
        feature = output / "features" / f"{name}.h5"
        relation = output / "relations" / f"{name}.h5"
        association = output / "relations" / "association" / name
        print(f"\n=== {name} ===", flush=True)

        if args.step in ("all", "timeline") and (args.force or not (timeline / "scene_build_report.json").is_file()):
            command = [py, str(PREP / "scripts" / "build_kitti360_timeline.py"),
                       "--dataset-root", str(raw), "--output-root", str(timeline),
                       "--scene-frames", "200", "--sample-stride", "5",
                       "--min-tail-frames", "100", "--split-file", str(SPLIT),
                       "--sequences", name]
            if args.force:
                command.append("--force")
            run(command, env, args.dry_run)

        if args.step in ("all", "predicates"):
            if args.force or not (predicates / "spatial_relations.json").is_file():
                command = [py, str(PREP / "scripts" / "build_kitti360_predicates.py"),
                           "--dataset-root", str(raw), "--timeline-root", str(timeline),
                           "--output-root", str(predicates)]
                if args.force:
                    command.append("--force")
                run(command, env, args.dry_run)
            if args.force or not (predicates / "temporal" / "temporal_relations.json").is_file():
                command = [py, str(PREP / "scripts" / "build_kitti360_temporal_predicates.py"),
                           "--timeline-root", str(timeline), "--predicate-root", str(predicates),
                           "--output-root", str(predicates / "temporal"),
                           "--minimum-joint-visible-steps-for-review", "3"]
                if args.force:
                    command.append("--force")
                run(command, env, args.dry_run)
            run([py, str(PREP / "scripts" / "validate_predicate_part.py"),
                 "--part-root", str(predicates), "--mode", "both"], env, args.dry_run)

        if args.step in ("all", "features") and (
            args.force or not (feature.is_file() and feature.with_suffix(".report.json").is_file())
        ):
            feature.parent.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
            command = [py, "-m", "IBP_KITTI360.ibp_model.prepare_features",
                       "--dataset-root", str(raw),
                       "--annotations", str(predicates / "sample_annotations.json"),
                       "--pointnet-checkpoint", str(WEIGHTS / "obj_enc.pth"),
                       "--output", str(feature), "--device", args.device]
            if args.force:
                command.append("--force")
            run(command, env, args.dry_run)

        if args.step in ("all", "relations") and (
            args.force or not (relation.is_file() and relation.with_suffix(".report.json").is_file())
        ):
            association.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
            command = [py, str(PREP / "scripts" / "build_kitti360_association_targets.py"),
                       "--timeline-root", str(timeline), "--predicate-root", str(predicates),
                       "--output-root", str(association)]
            if args.force:
                command.append("--force")
            run(command, env, args.dry_run)
            command = [py, "-m", "IBP_KITTI360.ibp_model.prepare_relation_index",
                       "--feature-file", str(feature),
                       "--feature-metadata", str(feature.with_suffix(".jsonl")),
                       "--spatial-relations", str(predicates / "spatial_relations.json"),
                       "--temporal-relations", str(predicates / "temporal" / "temporal_relations.json"),
                       "--association-candidates", str(association / "association_candidate.parquet"),
                       "--output", str(relation), "--full-multimodal-only"]
            if args.force:
                command.append("--force")
            run(command, env, args.dry_run)
    print(f"\nPreparation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
