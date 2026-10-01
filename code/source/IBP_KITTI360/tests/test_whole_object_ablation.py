from __future__ import annotations

import unittest

import torch

try:
    from KITTI360_IBP.ibp_model.losses import representation_losses
    from KITTI360_IBP.ibp_model.model import IBPK360Model
    from KITTI360_IBP.ibp_model.protocol import ModelConfig, SENSOR_MODALITIES
except ModuleNotFoundError:
    from IBP_KITTI360.ibp_model.losses import representation_losses
    from IBP_KITTI360.ibp_model.model import IBPK360Model
    from IBP_KITTI360.ibp_model.protocol import ModelConfig, SENSOR_MODALITIES


def compact_config(*, use_parts: bool) -> ModelConfig:
    return ModelConfig(
        dim=16,
        rgb_input_dim=8,
        lidar_input_dim=6,
        text_input_dim=5,
        anchor_input_dim=4,
        num_heads=4,
        fusion_layers=1,
        temporal_layers=1,
        dropout=0.0,
        use_parts=use_parts,
    )


def inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "rgb_tokens": torch.randn(batch, 4, 8),
        "lidar_tokens": torch.randn(batch, 3, 6),
        "lidar_anchor": torch.randn(batch, 4),
        "text_embedding": torch.randn(batch, 5),
        "modality_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


class WholeObjectAblationTest(unittest.TestCase):
    def test_legacy_configuration_remains_part_aware(self) -> None:
        current = compact_config(use_parts=True)
        legacy = current.to_dict()
        legacy.pop("use_parts")
        restored = ModelConfig(**legacy)
        self.assertTrue(restored.use_parts)
        self.assertEqual(
            set(IBPK360Model(current).state_dict()),
            set(IBPK360Model(restored).state_dict()),
        )

    def test_whole_object_encoder_keeps_all_modalities_without_part_modules(self) -> None:
        model = IBPK360Model(compact_config(use_parts=False))
        encoded = model.encode_objects(**inputs())
        self.assertEqual(model.config.enabled_modalities, SENSOR_MODALITIES)
        self.assertIsNone(model.part_interaction)
        self.assertFalse(hasattr(model.object_encoder, "queries"))
        self.assertEqual(encoded["fused_object"].shape, (2, 16))
        self.assertEqual(encoded["fused_parts"].shape, (2, 1, 16))

    def test_part_specific_losses_and_pair_feature_are_disabled(self) -> None:
        model = IBPK360Model(compact_config(use_parts=False))
        encoded = model.encode_objects(**inputs())
        losses = representation_losses(encoded)
        self.assertEqual(float(losses["diversity"]), 0.0)
        self.assertEqual(float(losses["part_alignment"]), 0.0)
        self.assertGreaterEqual(float(losses["object_alignment"]), 0.0)

        spatial = model.predict_spatial(
            encoded["fused_object"],
            encoded["fused_object"].flip(0),
            encoded["fused_parts"],
            encoded["fused_parts"].flip(0),
            torch.randn(2, model.config.spatial_geometry_dim),
        )
        temporal = model.predict_temporal(
            torch.randn(2, 5, 16),
            torch.randn(2, 5, 16),
            torch.ones(2, 5, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
            encoded["fused_parts"],
            encoded["fused_parts"].flip(0),
            torch.randn(2, model.config.temporal_geometry_dim),
        )
        self.assertEqual(spatial.shape, (2, 5))
        self.assertEqual(temporal.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
