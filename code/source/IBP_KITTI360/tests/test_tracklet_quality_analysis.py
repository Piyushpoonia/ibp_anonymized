from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

try:
    from KITTI360_IBP.ibp_model.analysis.tracklet_quality_v1.analyze import (
        analyze,
        align_predictions,
        load_predictions,
        quality_bin,
    )
except ModuleNotFoundError:
    from IBP_KITTI360.ibp_model.analysis.tracklet_quality_v1.analyze import (
        analyze,
        align_predictions,
        load_predictions,
        quality_bin,
    )


class Args:
    pass


class TrackletQualityAnalysisTest(unittest.TestCase):
    def test_quality_bin_boundaries(self) -> None:
        self.assertEqual(quality_bin(False, 5, False), "missing")
        self.assertEqual(quality_bin(True, 5, False), "perfect_5")
        self.assertEqual(quality_bin(True, 4, False), "good_4")
        self.assertEqual(quality_bin(True, 3, False), "partial_3")
        self.assertEqual(quality_bin(True, 5, True), "poor_identity")

    @staticmethod
    def write_prediction(
        path: Path,
        relation_ids: list[str],
        targets: np.ndarray,
        available: np.ndarray | None = None,
        reverse: bool = False,
    ) -> None:
        order = np.arange(len(relation_ids) - 1, -1, -1) if reverse else np.arange(len(relation_ids))
        logits = np.asarray(
            [[3.0, -2.0, -2.0], [-2.0, 3.0, -2.0], [-2.0, -2.0, 3.0], [2.0, 2.0, -2.0], [-2.0, -2.0, -2.0]],
            dtype=np.float32,
        )
        with h5py.File(path, "w") as handle:
            temporal = handle.create_group("temporal")
            temporal.create_dataset(
                "relation_ids",
                data=np.asarray([relation_ids[int(index)] for index in order], dtype="S32"),
            )
            temporal.create_dataset(
                "sequences",
                data=np.asarray(["sequence_0006"] * len(order), dtype="S32"),
            )
            temporal.create_dataset("raw_logits", data=logits[order])
            temporal.create_dataset("targets", data=targets[order])
            if available is not None:
                temporal.create_dataset("tracklet_available", data=available[order])

    @staticmethod
    def build_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
        relation_ids = [f"relation-{index}" for index in range(5)]
        targets = np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]],
            dtype=np.uint8,
        )
        teacher_path = root / "teacher.h5"
        predicted_path = root / "predicted.h5"
        TrackletQualityAnalysisTest.write_prediction(
            teacher_path, relation_ids, targets
        )
        TrackletQualityAnalysisTest.write_prediction(
            predicted_path,
            relation_ids,
            targets,
            available=np.asarray([1, 1, 1, 1, 0], dtype=np.uint8),
            reverse=True,
        )

        metadata: list[dict[str, object]] = []
        row_frames: list[int] = []
        row_track_ids: list[int] = []

        def add_row(frame: int, identity: str, track_id: int = -1) -> int:
            row = len(metadata)
            metadata.append(
                {
                    "feature_row": row,
                    "annotation_token": f"annotation-{row}",
                    "instance_token_supervision_only": identity,
                    "sequence": "sequence_0006",
                    "scene_token": "scene-a",
                    "raw_frame_index": frame,
                }
            )
            row_frames.append(frame)
            row_track_ids.append(track_id)
            return row

        source_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        frames_by_relation: list[list[int]] = []
        present_positions = [
            (set(range(5)), set(range(5))),
            ({1, 2, 3, 4}, set(range(5))),
            ({1, 2, 3}, set(range(5))),
            ({1, 2, 3, 4}, set(range(5))),
            ({2, 3}, set(range(5))),
        ]
        for relation_index, (source_present, target_present) in enumerate(present_positions):
            frames = [relation_index * 10 + position for position in range(5)]
            source_track = relation_index * 2
            target_track = source_track + 1
            local_source: list[int] = []
            local_target: list[int] = []
            for position, frame in enumerate(frames):
                source_row = add_row(
                    frame,
                    f"source-{relation_index}",
                    source_track if position in source_present else -1,
                )
                target_row = add_row(
                    frame,
                    f"target-{relation_index}",
                    target_track if position in target_present else -1,
                )
                local_source.append(source_row)
                local_target.append(target_row)
            if relation_index == 3:
                add_row(frames[0], "wrong-source", source_track)
                row_track_ids[local_source[0]] = -1
            source_rows.append(local_source)
            target_rows.append(local_target)
            frames_by_relation.append(frames)

        metadata_path = root / "sequence_0006.jsonl"
        metadata_path.write_text(
            "\n".join(json.dumps(value) for value in metadata) + "\n",
            encoding="utf-8",
        )
        relation_path = root / "sequence_0006.h5"
        with h5py.File(relation_path, "w") as handle:
            handle.attrs["schema_version"] = "IBP-K360-relation-index-v1.2.0"
            handle.attrs["sequence"] = "sequence_0006"
            temporal = handle.create_group("temporal")
            temporal.create_dataset("relation_ids", data=np.asarray(relation_ids, dtype="S32"))
            temporal.create_dataset("group_ids", data=np.asarray(["group-a"] * 5, dtype="S64"))
            temporal.create_dataset("targets", data=targets)
            temporal.create_dataset("source_rows", data=np.asarray(source_rows, dtype=np.int64))
            temporal.create_dataset("target_rows", data=np.asarray(target_rows, dtype=np.int64))
            temporal.create_dataset(
                "raw_frame_indices", data=np.asarray(frames_by_relation, dtype=np.int64)
            )
            temporal.create_dataset(
                "source_annotation_tokens",
                data=np.asarray(
                    [[metadata[row]["annotation_token"] for row in rows] for rows in source_rows],
                    dtype="S32",
                ),
            )
            temporal.create_dataset(
                "target_annotation_tokens",
                data=np.asarray(
                    [[metadata[row]["annotation_token"] for row in rows] for rows in target_rows],
                    dtype="S32",
                ),
            )

        track_path = root / "sequence_0006_tracks.h5"
        with h5py.File(track_path, "w") as handle:
            handle.attrs["schema_version"] = "IBP-K360-predicted-tracklets-v1.1.0"
            handle.attrs["sequence"] = "sequence_0006"
            handle.create_dataset("row_track_ids", data=np.asarray(row_track_ids, dtype=np.int64))
            handle.create_dataset("row_raw_frame_indices", data=np.asarray(row_frames, dtype=np.int64))
            handle.create_dataset(
                "row_annotation_tokens",
                data=np.asarray([value["annotation_token"] for value in metadata], dtype="S32"),
            )
            links = handle.create_group("accepted_links")
            links.create_dataset("source_rows", data=np.asarray([], dtype=np.int64))
            links.create_dataset("target_rows", data=np.asarray([], dtype=np.int64))
            links.create_dataset("confidence", data=np.asarray([], dtype=np.float32))
        return teacher_path, predicted_path, metadata_path, relation_path, track_path

    def test_analysis_stratifies_all_quality_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, predicted, metadata, relations, tracks = self.build_inputs(root)
            args = Args()
            args.teacher_predictions = teacher
            args.predicted_predictions = predicted
            args.feature_metadata = [metadata]
            args.relation_files = [relations]
            args.track_files = [tracks]
            args.output_root = root / "output"
            args.split_name = "validation"
            args.force = False
            result = analyze(args)
            self.assertEqual(
                result["quality_counts"],
                {
                    "perfect_5": 1,
                    "good_4": 1,
                    "partial_3": 1,
                    "poor_identity": 1,
                    "missing": 1,
                },
            )
            self.assertEqual(
                result["unavailable_reason_counts"],
                {"fewer_than_three_joint_steps": 1},
            )
            self.assertAlmostEqual(result["overall"]["tracklet_availability"], 0.8)
            self.assertTrue((args.output_root / "tracklet_quality_metrics.json").is_file())
            self.assertTrue((args.output_root / "tracklet_quality_records.csv.gz").is_file())
            self.assertTrue((args.output_root / "tracklet_quality_summary.csv").is_file())

    def test_prediction_alignment_reorders_extras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, predicted, _, _, _ = self.build_inputs(root)
            first, second = align_predictions(
                load_predictions(teacher), load_predictions(predicted)
            )
            self.assertEqual(first.relation_ids, second.relation_ids)
            self.assertEqual(
                second.extras["tracklet_available"].tolist(), [1, 1, 1, 1, 0]
            )


if __name__ == "__main__":
    unittest.main()
