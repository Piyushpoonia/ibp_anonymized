from pathlib import Path
import argparse, json

def indexed(folder, suffix):
    if not folder.exists():
        return {}
    result = {}
    for path in folder.glob(f"*{suffix}"):
        try:
            result[int(path.stem)] = path
        except ValueError:
            pass
    return result

def read_split(path):
    result = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields:
            parts = Path(fields[0]).parts
            result.add((parts[1], int(Path(fields[0]).stem)))
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

root = args.root.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)

split_root = root / "data_2d_semantics/train"
train = read_split(split_root / "2013_05_28_drive_train_frames.txt")
val = read_split(split_root / "2013_05_28_drive_val_frames.txt")

manifest_file = output / "synchronized_frames.jsonl"
summary = {"dataset_root": str(root), "sequences": {}, "totals": {}}
total_sync = total_windows = total_masks = 0

with manifest_file.open("w", encoding="utf-8") as stream:
    for seq_dir in sorted((root / "data_2d_raw").glob("*_sync")):
        seq = seq_dir.name
        left = indexed(seq_dir / "image_00/data_rect", ".png")
        right = indexed(seq_dir / "image_01/data_rect", ".png")
        lidar = indexed(root / f"data_3d_raw/{seq}/velodyne_points/data", ".bin")
        sem = indexed(root / f"data_2d_semantics/train/{seq}/image_00/semantic", ".png")
        inst = indexed(root / f"data_2d_semantics/train/{seq}/image_00/instance", ".png")
        oxts = indexed(root / f"data_poses/{seq}/oxts/data", ".txt")

        pose_file = root / f"data_poses/{seq}/cam0_to_world.txt"
        pose_ids = set()
        if pose_file.exists():
            for line in pose_file.read_text().splitlines():
                if line.strip():
                    pose_ids.add(int(line.split()[0]))

        frames = sorted(set(left) & set(lidar))
        windows = sum(
            all(frames[i + j] == frames[i] + j for j in range(5))
            for i in range(max(0, len(frames) - 4))
        )

        for frame in frames:
            key = (seq, frame)
            record = {
                "sequence": seq,
                "frame": frame,
                "rgb_left": str(left[frame].relative_to(root)),
                "rgb_right": str(right[frame].relative_to(root)) if frame in right else None,
                "lidar": str(lidar[frame].relative_to(root)),
                "semantic": str(sem[frame].relative_to(root)) if frame in sem else None,
                "instance": str(inst[frame].relative_to(root)) if frame in inst else None,
                "oxts": str(oxts[frame].relative_to(root)) if frame in oxts else None,
                "pose_file": str(pose_file.relative_to(root)),
                "exact_refined_pose": frame in pose_ids,
                "bbox_xml": f"data_3d_bboxes/train/{seq}.xml",
                "semantic_split": "train" if key in train else "val" if key in val else "unlabelled"
            }
            stream.write(json.dumps(record) + "\n")

        summary["sequences"][seq] = {
            "left_rgb": len(left), "lidar": len(lidar),
            "synchronized_frames": len(frames),
            "instance_masks": len(inst), "five_frame_windows": windows
        }
        total_sync += len(frames)
        total_windows += windows
        total_masks += len(inst)
        print(seq, summary["sequences"][seq], flush=True)

summary["totals"] = {
    "synchronized_frames": total_sync,
    "instance_masks": total_masks,
    "five_frame_windows": total_windows
}
(output / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["totals"], indent=2))
print("Saved:", manifest_file)
