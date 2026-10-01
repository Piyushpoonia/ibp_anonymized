import json
import tempfile
import unittest
from pathlib import Path

try:
    from KITTI360_IBP.ibp_model.evaluate_stage2_predcls import (
        ensure_output_below,
        load_and_validate_split,
        validate_checkpoint_metadata,
    )
    from KITTI360_IBP.ibp_model.protocol import (
        OBJECT_CLASSES,
        SPATIAL_PREDICATES,
        TEMPORAL_PREDICATES,
    )
except ModuleNotFoundError:
    from IBP_KITTI360.ibp_model.evaluate_stage2_predcls import (
        ensure_output_below,
        load_and_validate_split,
        validate_checkpoint_metadata,
    )
    from IBP_KITTI360.ibp_model.protocol import (
        OBJECT_CLASSES,
        SPATIAL_PREDICATES,
        TEMPORAL_PREDICATES,
    )


class Stage2PredCLSEvaluationTest(unittest.TestCase):
    @staticmethod
    def checkpoint() -> dict:
        return {
            "teacher_forced_tracklets": True,
            "final_predicted_tracklet_checkpoint": False,
            "object_classes": list(OBJECT_CLASSES),
            "spatial_predicates": list(SPATIAL_PREDICATES),
            "temporal_predicates": list(TEMPORAL_PREDICATES),
            "model": {},
            "model_config": {},
        }

    def test_accepts_teacher_forced_stage2_checkpoint_metadata(self):
        validate_checkpoint_metadata(self.checkpoint())

    def test_rejects_predicted_tracklet_checkpoint(self):
        checkpoint = self.checkpoint()
        checkpoint["final_predicted_tracklet_checkpoint"] = True
        with self.assertRaisesRegex(ValueError, "predicted-tracklet"):
            validate_checkpoint_metadata(checkpoint)

    def test_requires_exact_frozen_test_sequence_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "schema_version": "IBP-K360-sequence-split-v1.0.0",
                        "split_unit": "complete driving sequence",
                        "test": ["sequence_7", "sequence_9"],
                    }
                ),
                encoding="utf-8",
            )
            split, sequences = load_and_validate_split(
                split_path,
                [root / "sequence_7.h5", root / "sequence_9.h5"],
                [root / "sequence_7.h5", root / "sequence_9.h5"],
            )
            self.assertEqual(sequences, split["test"])
            with self.assertRaisesRegex(ValueError, "frozen held-out sequence order"):
                load_and_validate_split(
                    split_path,
                    [root / "sequence_9.h5", root / "sequence_7.h5"],
                    [root / "sequence_7.h5", root / "sequence_9.h5"],
                )

    def test_output_must_remain_inside_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            project.mkdir()
            self.assertEqual(
                ensure_output_below(project / "outputs" / "predcls", project),
                (project / "outputs" / "predcls").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "below the project root"):
                ensure_output_below(project.parent / "outside", project)


if __name__ == "__main__":
    unittest.main()
