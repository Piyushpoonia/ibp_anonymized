from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

try:
    from KITTI360_IBP.experiments.tracklet_recovery_v1.apply_frozen_held_out import (
        validate_frozen_selection,
    )
    from KITTI360_IBP.experiments.tracklet_recovery_v1.finalize_held_out import finalize
    from KITTI360_IBP.experiments.tracklet_recovery_v1.recover import (
        recover_track_file,
        select_links,
    )
except ModuleNotFoundError:
    from IBP_KITTI360.experiments.tracklet_recovery_v1.apply_frozen_held_out import (
        validate_frozen_selection,
    )
    from IBP_KITTI360.experiments.tracklet_recovery_v1.finalize_held_out import finalize
    from IBP_KITTI360.experiments.tracklet_recovery_v1.recover import (
        recover_track_file,
        select_links,
    )


class TrackletRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = np.asarray(
            [
                [5.0, 0.0, 0.0],
                [4.0, 3.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.mask = np.ones((2, 2), dtype=np.bool_)

    def test_one_way_fallback_recovers_unmatched_pair(self) -> None:
        links = select_links(self.logits, self.mask, "one_way", 0.0)
        self.assertEqual(
            [(link.source_index, link.target_index) for link in links],
            [(0, 0), (1, 1)],
        )
        self.assertFalse(links[0].fallback)
        self.assertTrue(links[1].fallback)

    def test_margin_can_reject_fallback_without_removing_mutual_link(self) -> None:
        links = select_links(self.logits, self.mask, "one_way", 3.1)
        self.assertEqual(len(links), 1)
        self.assertEqual((links[0].source_index, links[0].target_index), (0, 0))

    def test_recovered_links_remain_one_to_one(self) -> None:
        links = select_links(self.logits, self.mask, "all", -1.0)
        sources = [link.source_index for link in links]
        targets = [link.target_index for link in links]
        self.assertEqual(len(sources), len(set(sources)))
        self.assertEqual(len(targets), len(set(targets)))

    def test_recovered_file_preserves_stage3_track_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.h5"
            output_path = root / "recovered.h5"
            variable_int = h5py.vlen_dtype(np.dtype("int64"))
            variable_float = h5py.vlen_dtype(np.dtype("float32"))
            variable_byte = h5py.vlen_dtype(np.dtype("uint8"))
            with h5py.File(source_path, "w") as handle:
                handle.attrs["schema_version"] = "IBP-K360-predicted-tracklets-v1.1.0"
                handle.attrs["sequence"] = "sequence_0006"
                handle.attrs["source_checkpoint_sha256"] = "checkpoint-hash"
                handle.create_dataset("row_track_ids", data=np.arange(4, dtype=np.int64))
                handle.create_dataset(
                    "row_sample_indices", data=np.asarray([0, 0, 1, 1], dtype=np.int64)
                )
                handle.create_dataset(
                    "row_scene_tokens", data=np.asarray(["scene"] * 4, dtype="S32")
                )
                handle.create_group("accepted_links")
                groups = handle.create_group("association_groups")
                groups.create_dataset("shape", data=np.asarray([[2, 2]], dtype=np.int32))
                for name, dtype, value in (
                    ("source_rows", variable_int, np.asarray([0, 1], dtype=np.int64)),
                    ("target_rows", variable_int, np.asarray([2, 3], dtype=np.int64)),
                    ("raw_augmented_logits", variable_float, self.logits.reshape(-1)),
                    ("candidate_mask", variable_byte, self.mask.astype(np.uint8).reshape(-1)),
                ):
                    dataset = groups.create_dataset(name, (1,), dtype=dtype)
                    dataset[0] = value

            report = recover_track_file(
                source_path, output_path, "one_way", 0.0, False
            )
            with h5py.File(output_path, "r") as recovered:
                self.assertEqual(
                    recovered.attrs["schema_version"],
                    "IBP-K360-predicted-tracklets-v1.1.0",
                )
                self.assertEqual(recovered.attrs["recovery_mode"], "one_way")
                self.assertEqual(
                    recovered["row_track_ids"][:].tolist(), [0, 1, 0, 1]
                )
                self.assertEqual(
                    recovered["accepted_links/recovery_fallback"][:].tolist(),
                    [0, 1],
                )
            self.assertEqual(report["fallback_links"], 1)
            self.assertFalse(report["identity_labels_used_for_recovery"])

    def test_recovered_report_separates_mutual_and_fallback_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.h5"
            relation_path = root / "relation.h5"
            output_path = root / "recovered.h5"
            variable_int = h5py.vlen_dtype(np.dtype("int64"))
            variable_float = h5py.vlen_dtype(np.dtype("float32"))
            variable_byte = h5py.vlen_dtype(np.dtype("uint8"))
            with h5py.File(source_path, "w") as handle:
                handle.attrs["schema_version"] = "IBP-K360-predicted-tracklets-v1.1.0"
                handle.attrs["sequence"] = "2013_05_28_drive_0007_sync"
                handle.attrs["source_checkpoint_sha256"] = "checkpoint-hash"
                handle.create_dataset("row_track_ids", data=np.arange(4, dtype=np.int64))
                handle.create_dataset(
                    "row_sample_indices", data=np.asarray([0, 0, 1, 1], dtype=np.int64)
                )
                handle.create_dataset("row_scene_tokens", data=np.asarray(["scene"] * 4, dtype="S32"))
                handle.create_group("accepted_links")
                groups = handle.create_group("association_groups")
                groups.create_dataset("shape", data=np.asarray([[2, 2]], dtype=np.int32))
                for name, dtype, value in (
                    ("source_rows", variable_int, np.asarray([0, 1], dtype=np.int64)),
                    ("target_rows", variable_int, np.asarray([2, 3], dtype=np.int64)),
                    ("raw_augmented_logits", variable_float, self.logits.reshape(-1)),
                    ("candidate_mask", variable_byte, self.mask.astype(np.uint8).reshape(-1)),
                ):
                    dataset = groups.create_dataset(name, (1,), dtype=dtype)
                    dataset[0] = value
            with h5py.File(relation_path, "w") as handle:
                handle.attrs["schema_version"] = "IBP-K360-relation-index-v1.2.0"
                groups = handle.create_group("association")
                groups.create_dataset("shape", data=np.asarray([[2, 2]], dtype=np.int32))
                for name, value in (
                    ("source_rows", np.asarray([0, 1], dtype=np.int64)),
                    ("target_rows", np.asarray([2, 3], dtype=np.int64)),
                    ("source_targets", np.asarray([0, 1], dtype=np.int64)),
                    ("target_targets", np.asarray([0, 1], dtype=np.int64)),
                ):
                    dataset = groups.create_dataset(name, (1,), dtype=variable_int)
                    dataset[0] = value

            report = recover_track_file(
                source_path,
                output_path,
                "all",
                0.0,
                False,
                association_relation_path=relation_path,
            )
            self.assertEqual(report["association_correct"], 3)
            self.assertEqual(report["association_decisions"], 4)
            self.assertEqual(report["predicted_mutual_links"], 1)
            self.assertEqual(report["correct_mutual_links"], 1)
            self.assertEqual(report["fallback_links"], 1)
            self.assertEqual(report["correct_fallback_links"], 1)
            self.assertEqual(report["predicted_recovery_links"], 2)
            self.assertEqual(report["correct_recovery_links"], 2)

    def test_frozen_selection_rejects_a_changed_rule(self) -> None:
        summary = {
            "schema_version": "IBP-K360-tracklet-recovery-v1.0.0",
            "selection_split": "validation",
            "validation_sequence": "2013_05_28_drive_0006_sync",
            "selected_variant": {
                "variant": "all_margin_0p00",
                "mode": "all",
                "margin_threshold": 0.0,
            },
        }
        validate_frozen_selection(summary)
        summary["selected_variant"]["margin_threshold"] = 0.5
        with self.assertRaises(ValueError):
            validate_frozen_selection(summary)

    def test_final_metrics_record_recovery_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "final_test_metrics.json"
            manifest_path = root / "held_out_recovery_manifest.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "split": "held_out_test",
                        "publishable_final_result": True,
                        "test_sequences": [
                            "2013_05_28_drive_0007_sync",
                            "2013_05_28_drive_0009_sync",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "IBP-K360-tracklet-recovery-v1.0.0",
                        "validation_sequence": "2013_05_28_drive_0006_sync",
                        "frozen_variant": "all_margin_0p00",
                        "frozen_mode": "all",
                        "frozen_margin_threshold": 0.0,
                        "reports": [
                            {
                                "original_mutual_links": 4,
                                "fallback_links": 2,
                                "correct_fallback_links": 1,
                                "predicted_recovery_links": 6,
                                "correct_recovery_links": 5,
                                "true_forward_links": 8,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = finalize(
                SimpleNamespace(metrics=metrics_path, recovery_manifest=manifest_path)
            )
            recovery = result["tracklet_recovery"]
            self.assertEqual(recovery["variant"], "all_margin_0p00")
            self.assertEqual(recovery["fallback_links"], 2)
            self.assertAlmostEqual(recovery["recovery_link_precision"], 5 / 6)
            self.assertAlmostEqual(recovery["recovery_link_recall"], 5 / 8)
            repeated = finalize(
                SimpleNamespace(metrics=metrics_path, recovery_manifest=manifest_path)
            )
            self.assertEqual(repeated["tracklet_recovery"], recovery)


if __name__ == "__main__":
    unittest.main()
