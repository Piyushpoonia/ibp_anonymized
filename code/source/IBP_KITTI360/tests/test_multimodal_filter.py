from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from IBP_KITTI360.ibp_model.feature_data import FeatureShardDataset
from IBP_KITTI360.ibp_model.prepare_features import FeatureWriter
from IBP_KITTI360.ibp_model.prepare_relation_index import joint_multimodal_mask


def feature_row(mask: list[int], category: int) -> dict[str, np.ndarray]:
    return {
        "rgb_tokens": np.zeros((196, 768), dtype=np.float16),
        "lidar_tokens": np.zeros((128, 512), dtype=np.float16),
        "lidar_anchor": np.zeros(512, dtype=np.float16),
        "text_embedding": np.zeros(512, dtype=np.float16),
        "modality_mask": np.asarray(mask, dtype=np.uint8),
        "category_id": np.int16(category),
        "center_world": np.zeros(3, dtype=np.float32),
        "center_sensor": np.zeros(3, dtype=np.float32),
        "box_size": np.ones(3, dtype=np.float32),
    }


class MultimodalFilterTest(unittest.TestCase):
    def test_stage1_dataset_keeps_only_complete_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            writer = FeatureWriter(path, path.with_suffix(".jsonl"), force=True)
            writer.append(feature_row([1, 1, 1], 2), {"annotation_token": "full"})
            writer.append(feature_row([0, 1, 0], 3), {"annotation_token": "lidar"})
            writer.checkpoint(1)
            writer.close()
            dataset = FeatureShardDataset([path], require_all_modalities=True)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(int(dataset[0]["category_id"]), 2)
            dataset.close()

    def test_temporal_mask_requires_both_objects_at_each_step(self) -> None:
        masks = np.asarray(
            [[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1]],
            dtype=np.uint8,
        )
        source = [0, 0, 0, 0, 0]
        target = [1, 2, 1, 3, 4]
        self.assertEqual(joint_multimodal_mask(source, target, masks), [1, 0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
