from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from IBP_KITTI360.ibp_model.prepare_features import object_crop


class ObjectCropTest(unittest.TestCase):
    def setUp(self) -> None:
        pixels = np.full((12, 12, 3), 127, dtype=np.uint8)
        self.image = Image.fromarray(pixels)
        self.annotation = {
            "projected_bbox_xyxy": [2.0, 3.0, 10.0, 11.0],
            "camera_depth_m": 8.0,
            "combined_instance_id": 11001,
            "rgb_usable": False,
        }

    def test_projected_box_is_used_when_mask_id_does_not_match(self) -> None:
        mask = np.full((12, 12), 11115, dtype=np.uint16)
        crop, source = object_crop(self.image, mask, self.annotation)
        self.assertIsNotNone(crop)
        self.assertEqual(source, "projected_3d_box")
        self.assertTrue((np.asarray(crop) == 127).all())

    def test_matching_mask_refines_the_projected_crop(self) -> None:
        mask = np.zeros((12, 12), dtype=np.uint16)
        mask[4:8, 4:8] = 11001
        crop, source = object_crop(self.image, mask, self.annotation)
        self.assertIsNotNone(crop)
        self.assertEqual(source, "matched_instance_mask")
        values = np.asarray(crop)
        self.assertTrue((values[1:5, 2:6] == 127).all())
        self.assertTrue((values[0, 0] == 0).all())


if __name__ == "__main__":
    unittest.main()
