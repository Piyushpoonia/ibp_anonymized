"""Train the frozen three-stage IBP-K360 model from prepared shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "code" / "source"
SPLIT = SOURCE / "IBP_KITTI360" / "splits" / "kitti360_ibp_sequence_split_v1.json"
RECOVERY_SELECTION = SOURCE / "IBP_KITTI360" / "config" / "frozen_recovery_selection.json"


def files(root: Path, folder: str, names: list[str], suffix: str = ".h5") -> list[str]:
    return [str(root / folder / f"{name}{suffix}") for name in names]


def require(paths: list[str | Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing prerequisite files:\n" + "\n".join(missing[:12]))


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=ROOT / "PREPARED_DATASET")
    parser.add_argument("--output-root", type=Path, default=ROOT / "RUNS" / "seed_42")
    parser.add_argument(
        "--phase",
        choices=("all", "stage1", "stage2", "tracks", "stage3", "evaluate", "recover"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepared = args.prepared_root.resolve()
    output = args.output_root.resolve()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train = split["train"]
    val = split["validation"]
    test = split["test"]
    train_features = files(prepared, "features", train)
    train_relations = files(prepared, "relations", train)
    val_features = files(prepared, "features", val)
    val_relations = files(prepared, "relations", val)
    test_features = files(prepared, "features", test)
    test_relations = files(prepared, "relations", test)
    stage1 = output / "stage1"
    stage2 = output / "stage2_warmup"
    stage3 = output / "stage3_predicted"
    stage1_best = stage1 / "checkpoints" / "stage1_best_macro_f1.pt"
    stage2_best = stage2 / "checkpoints" / "stage2_warmup_best.pt"
    stage3_best = stage3 / "checkpoints" / "stage3_best.pt"
    tracks = output / "predicted_tracklets"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = str(args.seed)
    py = sys.executable

    if not args.dry_run:
        require(train_features + val_features + test_features + train_relations + val_relations + test_relations)
        output.mkdir(parents=True, exist_ok=True)

    if args.phase in ("all", "stage1"):
        if not (stage1 / "stage1_result.json").is_file():
            command = [py, "-u", "-m", "IBP_KITTI360.ibp_model.train_stage1",
                       "--train-features", *train_features, "--val-features", *val_features,
                       "--output-root", str(stage1), "--epochs", "40", "--batch-size", "32",
                       "--seed", str(args.seed), "--full-multimodal-only", "--device", args.device]
            if (stage1 / "checkpoints" / "stage1_last.pt").is_file():
                command.append("--resume")
            run(command, env, args.dry_run)
        else:
            print("Stage 1 already complete; preserving checkpoint.")

    if args.phase in ("all", "stage2"):
        if not args.dry_run:
            require([stage1_best])
        if not (stage2 / "stage2_warmup_result.json").is_file():
            command = [py, "-u", "-m", "IBP_KITTI360.ibp_model.train_stage2_warmup",
                       "--train-features", *train_features, "--train-relations", *train_relations,
                       "--val-features", *val_features, "--val-relations", *val_relations,
                       "--stage1-checkpoint", str(stage1_best), "--output-root", str(stage2),
                       "--epochs", "30", "--batch-size", "16", "--steps-per-epoch", "1000",
                       "--seed", str(args.seed), "--full-multimodal-only", "--device", args.device]
            if (stage2 / "checkpoints" / "stage2_warmup_last.pt").is_file():
                command.append("--resume")
            run(command, env, args.dry_run)
        else:
            print("Stage 2 already complete; preserving checkpoint.")

    def make_tracks(names: list[str]) -> None:
        if not args.dry_run:
            require([stage2_best])
        for name in names:
            target = tracks / f"{name}.h5"
            report = tracks / f"{name}.report.json"
            if target.is_file() and report.is_file():
                print(f"Tracklet shard already complete: {name}")
                continue
            if not args.dry_run:
                tracks.mkdir(parents=True, exist_ok=True)
            run([py, "-u", "-m", "IBP_KITTI360.ibp_model.predicted_tracklets",
                 "--feature-file", str(prepared / "features" / f"{name}.h5"),
                 "--feature-metadata", str(prepared / "features" / f"{name}.jsonl"),
                 "--relation-file", str(prepared / "relations" / f"{name}.h5"),
                 "--timeline-root", str(prepared / "timeline" / "parts" / name),
                 "--checkpoint", str(stage2_best), "--output", str(target),
                 "--device", args.device], env, args.dry_run)

    if args.phase in ("all", "tracks", "stage3"):
        make_tracks(train + val)

    if args.phase in ("all", "stage3"):
        if not args.dry_run:
            require([stage2_best] + files(tracks, "", train + val))
        if not (stage3 / "stage3_result.json").is_file():
            command = [py, "-u", "-m", "IBP_KITTI360.ibp_model.train_stage3_predicted",
                       "--train-features", *train_features, "--train-relations", *train_relations,
                       "--train-tracks", *files(tracks, "", train),
                       "--val-features", *val_features, "--val-relations", *val_relations,
                       "--val-tracks", *files(tracks, "", val),
                       "--stage2-checkpoint", str(stage2_best), "--output-root", str(stage3),
                       "--epochs", "10", "--steps-per-epoch", "1000", "--batch-size", "16",
                       "--seed", str(args.seed), "--device", args.device]
            if (stage3 / "checkpoints" / "stage3_last.pt").is_file():
                command.append("--resume")
            run(command, env, args.dry_run)
        else:
            print("Stage 3 already complete; preserving checkpoint.")

    if args.phase in ("all", "evaluate"):
        if not args.dry_run:
            require([stage3_best])
        make_tracks(test)
        final = output / "final_test"
        if not (final / "final_test_metrics.json").is_file():
            run([py, "-u", "-m", "IBP_KITTI360.ibp_model.evaluate_stage3",
                 "--test-features", *test_features,
                 "--test-metadata", *files(prepared, "features", test, ".jsonl"),
                 "--test-relations", *test_relations,
                 "--test-tracks", *files(tracks, "", test),
                 "--checkpoint", str(stage3_best), "--output-root", str(final),
                 "--device", args.device], env, args.dry_run)

    if args.phase in ("all", "recover"):
        if not args.dry_run:
            require([stage3_best, RECOVERY_SELECTION])
        if args.phase == "recover":
            make_tracks(test)
        recovery = output / "tracklet_recovery"
        recovered_tracks = files(recovery / "tracks" / "all_margin_0p00", "", test)
        manifest = recovery / "held_out_recovery_manifest.json"
        if not manifest.is_file():
            run([py, "-u", "-m", "IBP_KITTI360.experiments.tracklet_recovery_v1.apply_frozen_held_out",
                 "--validation-summary", str(RECOVERY_SELECTION),
                 "--source-tracks", *files(tracks, "", test),
                 "--relation-files", *test_relations,
                 "--output-root", str(recovery)], env, args.dry_run)
        recovered = output / "final_test_recovered"
        metrics = recovered / "final_test_metrics.json"
        if not metrics.is_file():
            run([py, "-u", "-m", "IBP_KITTI360.ibp_model.evaluate_stage3",
                 "--test-features", *test_features,
                 "--test-metadata", *files(prepared, "features", test, ".jsonl"),
                 "--test-relations", *test_relations,
                 "--test-tracks", *recovered_tracks,
                 "--checkpoint", str(stage3_best), "--output-root", str(recovered),
                 "--device", args.device], env, args.dry_run)
        if not args.dry_run:
            require([metrics, manifest])
        run([py, "-u", "-m", "IBP_KITTI360.experiments.tracklet_recovery_v1.finalize_held_out",
             "--metrics", str(metrics), "--recovery-manifest", str(manifest)], env, args.dry_run)
    print(f"\nRun complete: {output}", flush=True)


if __name__ == "__main__":
    main()
