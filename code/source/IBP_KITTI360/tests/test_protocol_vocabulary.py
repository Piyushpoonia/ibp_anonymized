from __future__ import annotations

import unittest

try:
    from KITTI360_IBP.ibp_model.protocol import (
        ModelConfig,
        OBJECT_CLASSES,
        validate_protocol,
    )
except ModuleNotFoundError:
    from IBP_KITTI360.ibp_model.protocol import (
        ModelConfig,
        OBJECT_CLASSES,
        validate_protocol,
    )


EXPECTED_CLASSES = (
    "building",
    "garage",
    "car",
    "truck",
    "trailer",
    "caravan",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "bigPole",
    "smallPole",
    "trafficLight",
    "trafficSign",
    "lamp",
    "trashbin",
    "vendingmachine",
    "box",
    "stop",
    "bridge",
    "tunnel",
    "train",
    "bus",
    "unknownConstruction",
    "unknownVehicle",
    "unknownObject",
)


class ProtocolVocabularyTest(unittest.TestCase):
    def test_full_kitti360_box_vocabulary_is_frozen_in_semantic_order(self) -> None:
        self.assertEqual(OBJECT_CLASSES, EXPECTED_CLASSES)
        self.assertEqual(len(OBJECT_CLASSES), 26)
        self.assertEqual(len(set(OBJECT_CLASSES)), len(OBJECT_CLASSES))

    def test_obsolete_18_class_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen ontology"):
            validate_protocol(ModelConfig(num_object_classes=18))


if __name__ == "__main__":
    unittest.main()
