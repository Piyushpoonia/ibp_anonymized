from __future__ import annotations

import unittest

import torch

try:
    from KITTI360_IBP.ibp_model.model import IBPK360Model
    from KITTI360_IBP.ibp_model.protocol import ModelConfig, validate_protocol
except ModuleNotFoundError:
    from IBP_KITTI360.ibp_model.model import IBPK360Model
    from IBP_KITTI360.ibp_model.protocol import ModelConfig, validate_protocol


class RGBOnlyAblationTest(unittest.TestCase):
    def test_disabled_modalities_cannot_change_object_representation(self):
        torch.manual_seed(42)
        config = ModelConfig(
            dim=32,
            num_heads=4,
            fusion_layers=1,
            temporal_layers=1,
            enabled_modalities=("rgb",),
        )
        model = IBPK360Model(config).eval()
        inputs = {
            "rgb_tokens": torch.randn(2, 196, 768),
            "lidar_tokens": torch.randn(2, 128, 512),
            "lidar_anchor": torch.randn(2, 512),
            "text_embedding": torch.randn(2, 512),
            "modality_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["lidar_tokens"] = torch.randn_like(inputs["lidar_tokens"]) * 1000
        changed["lidar_anchor"] = torch.randn_like(inputs["lidar_anchor"]) * 1000
        changed["text_embedding"] = torch.randn_like(inputs["text_embedding"]) * 1000

        with torch.inference_mode():
            first = model.encode_objects(**inputs)
            second = model.encode_objects(**changed)

        expected_mask = torch.tensor(
            [[True, False, False], [True, False, False]]
        )
        self.assertTrue(torch.equal(first["modality_mask"], expected_mask))
        self.assertTrue(torch.allclose(first["fused_object"], second["fused_object"]))
        self.assertTrue(torch.allclose(first["fused_parts"], second["fused_parts"]))

    def test_modality_configuration_is_checkpoint_serializable(self):
        config = ModelConfig(enabled_modalities=("rgb",))
        restored = ModelConfig(**config.to_dict())
        validate_protocol(restored)
        self.assertEqual(restored.enabled_modalities, ("rgb",))

    def test_empty_modality_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_protocol(ModelConfig(enabled_modalities=()))


if __name__ == "__main__":
    unittest.main()
