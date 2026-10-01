from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from IBP_KITTI360.ibp_model.prepare_features import FeatureWriter


def empty_row() -> dict[str, np.ndarray]:
    return {
        "rgb_tokens": np.zeros((196, 768), dtype=np.float16),
        "lidar_tokens": np.zeros((128, 512), dtype=np.float16),
        "lidar_anchor": np.zeros(512, dtype=np.float16),
        "text_embedding": np.zeros(512, dtype=np.float16),
        "modality_mask": np.asarray([0, 1, 0], dtype=np.uint8),
        "category_id": np.int16(2),
        "center_world": np.zeros(3, dtype=np.float32),
        "center_sensor": np.zeros(3, dtype=np.float32),
        "box_size": np.ones(3, dtype=np.float32),
    }


class FeatureWriterRecoveryTest(unittest.TestCase):
    def test_rolls_back_an_uncommitted_partial_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.h5"
            metadata_path = feature_path.with_suffix(".jsonl")
            writer = FeatureWriter(feature_path, metadata_path, force=True)
            writer.append(empty_row(), {"annotation_token": "committed"})
            writer.checkpoint(10)
            writer.close()

            with h5py.File(feature_path, "a") as feature_file:
                category = feature_file["category_id"]
                category.resize(2, axis=0)
                category[1] = 3
            with metadata_path.open("a", encoding="utf-8") as stream:
                stream.write('{"feature_row":1,"annotation_token":"partial"}\n')

            recovered = FeatureWriter(feature_path, metadata_path, force=False)
            self.assertEqual(recovered.rows, 1)
            recovered.close()
            self.assertEqual(len(metadata_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
