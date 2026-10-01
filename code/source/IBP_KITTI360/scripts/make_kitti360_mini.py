from __future__ import annotations

import argparse
import io
import json
import os
import stat
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


RECORD_KEYS = ("rgb_left", "rgb_right", "lidar", "semantic", "instance", "oxts")

# Largest synchronized sequences first. The first three are expected to exceed 100 GiB.
SEQUENCE_ORDER = (
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0010_sync",
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0003_sync",
)


def contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_records(manifest: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            grouped[record["sequence"]].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["frame"])
    return grouped


def dynamic_box_counts(xml_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not xml_path.is_file():
        return counts
    for child in ET.parse(xml_path).getroot():
        if child.find("transform") is None:
            continue
        timestamp = int(child.findtext("timestamp", "-1"))
        if timestamp >= 0:
            counts[timestamp] += 1
    return counts


def record_sources(dataset: Path, record: dict) -> list[Path]:
    sources = [dataset / record[key] for key in RECORD_KEYS if record.get(key)]
    frame_name = f"{record['frame']:010d}.png"
    semantic_root = (
        dataset
        / "data_2d_semantics"
        / "train"
        / record["sequence"]
        / "image_00"
    )
    sources.extend(
        [
            semantic_root / "confidence" / frame_name,
            semantic_root / "semantic_rgb" / frame_name,
        ]
    )
    return sources


def normalized_source(source: Path, dataset: Path) -> tuple[Path, Path]:
    # Lexical normalization avoids expensive filesystem-wide resolve calls.
    normalized = Path(os.path.abspath(source))
    try:
        relative = normalized.relative_to(dataset)
    except ValueError as error:
        raise RuntimeError(f"Refusing to read outside dataset root: {source}") from error
    if normalized.is_symlink():
        raise RuntimeError(f"Refusing symbolic-link input: {normalized}")
    return normalized, relative


def add_file(
    output: tarfile.TarFile,
    source: Path,
    dataset: Path,
    archive_root: Path,
    added: set[Path],
) -> int:
    normalized, relative = normalized_source(source, dataset)
    try:
        source_stat = normalized.stat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(source_stat.st_mode) or normalized in added:
        return 0
    output.add(
        normalized,
        arcname=(archive_root / relative).as_posix(),
        recursive=False,
    )
    added.add(normalized)
    return source_stat.st_size


def add_tree(
    output: tarfile.TarFile,
    source: Path,
    dataset: Path,
    archive_root: Path,
    added: set[Path],
) -> int:
    if not source.is_dir():
        return 0
    return sum(
        add_file(output, path, dataset, archive_root, added)
        for path in source.rglob("*")
        if path.is_file()
    )


def add_bytes(output: tarfile.TarFile, arcname: str, content: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(content)
    info.mtime = int(time.time())
    output.addfile(info, io.BytesIO(content))


def sequence_metadata(dataset: Path, sequence: str) -> list[Path]:
    return [
        dataset / "data_3d_bboxes" / "train" / f"{sequence}.xml",
        dataset / "data_2d_raw" / sequence / "image_00" / "timestamps.txt",
        dataset / "data_2d_raw" / sequence / "image_01" / "timestamps.txt",
        dataset / "data_3d_raw" / sequence / "velodyne_points" / "timestamps.txt",
        dataset / "data_poses" / sequence / "poses.txt",
        dataset / "data_poses" / sequence / "cam0_to_world.txt",
        dataset / "data_poses" / sequence / "oxts" / "timestamps.txt",
        dataset / "data_poses" / sequence / "oxts" / "dataformat.txt",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--minimum-gib", type=float, default=100.0)
    args = parser.parse_args()

    dataset = args.dataset_root.resolve()
    project = args.project_root.resolve()
    expected_project = Path("/home/shrutim_iitp/piyush/IBP_KITTI360").resolve()
    if project != expected_project:
        raise RuntimeError(f"Output must remain inside {expected_project}")
    if not contained(args.manifest, project):
        raise RuntimeError("Manifest must be inside the project directory")
    if args.minimum_gib < 1:
        raise RuntimeError("--minimum-gib must be at least 1")

    grouped = load_records(args.manifest)
    minimum_bytes = int(args.minimum_gib * (1024 ** 3))
    subset_name = f"KITTI360_development_{args.minimum_gib:g}GiB"
    archive_root = Path(subset_name)
    export_root = project / "mini_export"
    export_root.mkdir(parents=True, exist_ok=True)
    archive = export_root / f"{subset_name}.tar.gz"
    partial_archive = export_root / f".{subset_name}.tar.gz.partial"
    selection_path = export_root / f"{subset_name}_selection.json"

    selected: dict[str, list[dict]] = {}
    payload_bytes = 0
    added: set[Path] = set()
    selection = {
        "minimum_payload_gib": args.minimum_gib,
        "selection_policy": "largest complete synchronized sequences first",
        "sequences": {},
    }

    print("Opening direct compressed archive; no preliminary size scan is used.", flush=True)
    with tarfile.open(partial_archive, "w:gz", compresslevel=1) as output:
        for sequence in SEQUENCE_ORDER:
            records = grouped.get(sequence)
            if not records:
                continue
            print(f"Starting complete sequence {sequence}: {len(records)} frames", flush=True)
            dynamic = dynamic_box_counts(
                dataset / "data_3d_bboxes" / "train" / f"{sequence}.xml"
            )
            sequence_bytes = 0
            for index, record in enumerate(records, start=1):
                for source in record_sources(dataset, record):
                    sequence_bytes += add_file(
                        output, source, dataset, archive_root, added
                    )
                if index % 100 == 0 or index == len(records):
                    print(
                        f"{sequence}: archived {index}/{len(records)} frame records "
                        f"({sequence_bytes / (1024 ** 3):.2f} GiB source)",
                        flush=True,
                    )

            for source in sequence_metadata(dataset, sequence):
                add_file(output, source, dataset, archive_root, added)
            semantic_3d_bytes = add_tree(
                output,
                dataset / "data_3d_semantics" / "train" / sequence,
                dataset,
                archive_root,
                added,
            )

            selected[sequence] = records
            payload_bytes += sequence_bytes
            frames = [record["frame"] for record in records]
            selection["sequences"][sequence] = {
                "start_frame": min(frames),
                "end_frame": max(frames),
                "selected_frames": len(records),
                "frames_with_instance_masks": sum(
                    bool(record.get("instance")) for record in records
                ),
                "dynamic_box_observations": sum(dynamic[frame] for frame in frames),
                "synchronized_payload_gib": sequence_bytes / (1024 ** 3),
                "additional_3d_semantics_gib": semantic_3d_bytes / (1024 ** 3),
            }
            print(
                f"Cumulative synchronized payload: {payload_bytes / (1024 ** 3):.2f} GiB",
                flush=True,
            )
            if payload_bytes >= minimum_bytes:
                break

        if payload_bytes < minimum_bytes:
            raise RuntimeError(
                f"Available synchronized payload was only {payload_bytes / (1024 ** 3):.2f} GiB"
            )

        shared_metadata = [
            dataset / "README.md",
            dataset / "data_2d_semantics" / "train" / "2013_05_28_drive_train_frames.txt",
            dataset / "data_2d_semantics" / "train" / "2013_05_28_drive_val_frames.txt",
            dataset / "data_3d_semantics" / "train" / "2013_05_28_drive_train.txt",
            dataset / "data_3d_semantics" / "train" / "2013_05_28_drive_val.txt",
        ]
        shared_metadata.extend(
            path for path in (dataset / "calibration").iterdir() if path.is_file()
        )
        for source in shared_metadata:
            add_file(output, source, dataset, archive_root, added)

        manifest_content = "".join(
            json.dumps(record) + "\n"
            for sequence in selected
            for record in selected[sequence]
        ).encode("utf-8")
        selection["selected_sequence_count"] = len(selected)
        selection["selected_frame_records"] = sum(
            len(records) for records in selected.values()
        )
        selection["selected_payload_gib"] = payload_bytes / (1024 ** 3)
        selection_content = json.dumps(selection, indent=2).encode("utf-8")
        add_bytes(
            output,
            (archive_root / "subset_manifest.jsonl").as_posix(),
            manifest_content,
        )
        add_bytes(
            output,
            (archive_root / "subset_selection.json").as_posix(),
            selection_content,
        )

    partial_archive.replace(archive)
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Archive: {archive}")
    print(f"Archive size: {archive.stat().st_size / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
