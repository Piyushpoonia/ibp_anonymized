from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def invoke(self, script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_preparation_dry_run_has_all_nine_shards(self) -> None:
        result = self.invoke(ROOT / "code" / "data_preparation" / "prepare.py", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("=== 2013_05_28_drive_"), 9)
        self.assertEqual(result.stdout.count("-m IBP_KITTI360.ibp_model.prepare_features"), 9)
        self.assertEqual(result.stdout.count("-m IBP_KITTI360.ibp_model.prepare_relation_index"), 9)

    def test_training_dry_run_preserves_stage_order(self) -> None:
        result = self.invoke(ROOT / "code" / "source" / "train.py", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        names = (
            "ibp_model.train_stage1",
            "ibp_model.train_stage2_warmup",
            "ibp_model.train_stage3_predicted",
            "ibp_model.evaluate_stage3",
        )
        indices = [output.index(name) for name in names]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(output.count("ibp_model.predicted_tracklets"), 9)
        self.assertIn("tracklet_recovery_v1.apply_frozen_held_out", output)
        self.assertIn("tracklet_recovery_v1.finalize_held_out", output)

    def test_preparation_rejects_missing_raw_dataset(self) -> None:
        result = self.invoke(ROOT / "code" / "data_preparation" / "prepare.py", "--step", "timeline")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Incomplete KITTI-360 raw dataset", result.stderr)


if __name__ == "__main__":
    unittest.main()
