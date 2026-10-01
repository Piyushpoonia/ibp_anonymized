from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    from KITTI360_IBP.ibp_model.evaluate_stage3 import relation_metadata
    from KITTI360_IBP.ibp_model.predicted_relation_data import (
        PredictedTemporalRelationDataset,
        _orientation,
        predicted_temporal_geometry,
    )
    from KITTI360_IBP.ibp_model.predicted_tracklets import mutual_matches
except ModuleNotFoundError:
    from IBP_KITTI360.ibp_model.evaluate_stage3 import relation_metadata
    from IBP_KITTI360.ibp_model.predicted_relation_data import (
        PredictedTemporalRelationDataset,
        _orientation,
        predicted_temporal_geometry,
    )
    from IBP_KITTI360.ibp_model.predicted_tracklets import mutual_matches


class PredictedTrackletTest(unittest.TestCase):
    def test_planar_orientation_does_not_depend_on_numpy_cross(self):
        origin = np.asarray([0.0, 0.0])
        right = np.asarray([1.0, 0.0])
        up = np.asarray([0.0, 1.0])
        self.assertEqual(_orientation(origin, right, up), 1.0)
        self.assertEqual(_orientation(origin, up, right), -1.0)

    def test_mutual_matching_rejects_dustbins_and_keeps_one_to_one_links(self):
        augmented = torch.tensor(
            [
                [5.0, 0.0, -2.0],
                [0.0, 4.0, -2.0],
                [-2.0, -2.0, 0.0],
            ]
        )
        source, target, links = mutual_matches(
            augmented, torch.ones((2, 2), dtype=torch.bool)
        )
        self.assertEqual(source.tolist(), [0, 1])
        self.assertEqual(target.tolist(), [0, 1])
        self.assertEqual(links, [(0, 0), (1, 1)])

    def test_predicted_geometry_uses_the_predicted_trajectory(self):
        source = np.asarray([[value, 0.0, 0.0] for value in range(5)])
        target = np.asarray([[10.0, 0.0, 0.0] for _ in range(5)])
        timestamps = np.arange(5, dtype=np.int64) * 500_000_000
        geometry = predicted_temporal_geometry(
            source, target, timestamps, np.ones(5, dtype=np.bool_)
        )
        self.assertEqual(geometry.shape, (24,))
        self.assertAlmostEqual(float(geometry[5]), -4.0)
        self.assertAlmostEqual(float(geometry[8]), 2.0)
        self.assertAlmostEqual(float(geometry[16]), 2.0)
        self.assertAlmostEqual(float(geometry[19]), 8.0)

    def test_tracklet_availability_comes_from_predicted_track_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "sequence.h5"
            relation_path = root / "relations.h5"
            track_path = root / "tracks.h5"
            with h5py.File(feature_path, "w"):
                pass
            with h5py.File(relation_path, "w") as relation_file:
                relation_file.attrs["schema_version"] = "IBP-K360-relation-index-v1.2.0"
                temporal = relation_file.create_group("temporal")
                temporal.create_dataset("targets", data=np.zeros((1, 3), dtype=np.uint8))
                temporal.create_dataset(
                    "source_rows", data=np.asarray([[0, 2, 4, 6, 8]], dtype=np.int64)
                )
                temporal.create_dataset(
                    "target_rows", data=np.asarray([[1, 3, 5, 7, 9]], dtype=np.int64)
                )
                temporal.create_dataset(
                    "raw_frame_indices", data=np.asarray([[0, 5, 10, 15, 20]], dtype=np.int64)
                )
            with h5py.File(track_path, "w") as track_file:
                track_file.attrs["schema_version"] = (
                    "IBP-K360-predicted-tracklets-v1.1.0"
                )
                track_file.create_dataset(
                    "row_track_ids", data=np.asarray([0, 1] * 5, dtype=np.int64)
                )
                track_file.create_dataset(
                    "row_raw_frame_indices",
                    data=np.repeat(np.asarray([0, 5, 10, 15, 20]), 2),
                )
                track_file.create_dataset(
                    "timeline_raw_frame_indices",
                    data=np.asarray([0, 5, 10, 15, 20]),
                )
                track_file.create_dataset(
                    "timeline_timestamps_ns",
                    data=np.arange(5, dtype=np.int64) * 500_000_000,
                )
            dataset = PredictedTemporalRelationDataset(
                [feature_path], [relation_path], [track_path]
            )
            self.assertEqual(dataset.eligible_indices, [0])
            source_rows, target_rows, available = dataset.shards[0].predicted_rows(0)
            self.assertTrue(available)
            self.assertEqual(source_rows.tolist(), [0, 2, 4, 6, 8])
            self.assertEqual(target_rows.tolist(), [1, 3, 5, 7, 9])
            dataset.close()

    def test_final_metadata_keeps_relation_object_and_frame_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relations.h5"
            with h5py.File(path, "w") as relation_file:
                relation_file.attrs["schema_version"] = (
                    "IBP-K360-relation-index-v1.2.0"
                )
                relation_file.attrs["sequence"] = "sequence_0007"
                spatial = relation_file.create_group("spatial")
                spatial.create_dataset("targets", data=np.zeros((1, 5), dtype=np.uint8))
                spatial.create_dataset("relation_ids", data=np.asarray([b"relation-a"]))
                spatial.create_dataset("group_ids", data=np.asarray([b"sample-a"]))
                spatial.create_dataset(
                    "source_annotation_tokens", data=np.asarray([b"subject-a"])
                )
                spatial.create_dataset(
                    "target_annotation_tokens", data=np.asarray([b"object-a"])
                )
                spatial.create_dataset(
                    "raw_frame_indices", data=np.asarray([123], dtype=np.int64)
                )
            metadata = relation_metadata([path], "spatial")
            self.assertEqual(metadata["relation_ids"], ["relation-a"])
            self.assertEqual(metadata["sequences"], ["sequence_0007"])
            self.assertEqual(metadata["source_annotation_tokens"].tolist(), ["subject-a"])
            self.assertEqual(metadata["target_annotation_tokens"].tolist(), ["object-a"])
            self.assertEqual(metadata["raw_frame_indices"].tolist(), [123])


if __name__ == "__main__":
    unittest.main()
