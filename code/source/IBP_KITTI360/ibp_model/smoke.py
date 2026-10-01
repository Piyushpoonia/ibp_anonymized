from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .losses import (
    association_cross_entropy,
    multilabel_focal_loss,
    representation_losses,
    temporal_consistency,
)
from .model import IBPK360Model
from .pointnet import load_geometry_pointnet
from .protocol import ModelConfig, SPATIAL_PREDICATES, TEMPORAL_PREDICATES


def random_object_inputs(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rgb_tokens": torch.randn(batch, 196, 768, device=device),
        "lidar_tokens": torch.randn(batch, 128, 512, device=device),
        "lidar_anchor": torch.randn(batch, 512, device=device),
        "text_embedding": torch.randn(batch, 512, device=device),
        "modality_mask": torch.tensor(
            [[1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1]],
            dtype=torch.bool,
            device=device,
        )[:batch],
    }


def run_smoke(device_name: str, checkpoint: Path | None) -> dict[str, object]:
    torch.manual_seed(42)
    device = torch.device(device_name)
    config = ModelConfig()
    model = IBPK360Model(config).to(device).train()
    encoded = model.encode_objects(**random_object_inputs(4, device))
    rep = representation_losses(encoded)

    pair_geometry = torch.randn(4, 3, config.association_geometry_dim, device=device)
    candidate_mask = torch.ones(4, 3, dtype=torch.bool, device=device)
    candidate_mask[3, 0] = False
    association = model.association(
        encoded["fused_object"], encoded["fused_object"][:3], pair_geometry, candidate_mask
    )
    source_target = torch.tensor([0, 1, 2, 3], device=device)
    target_source = torch.tensor([0, 1, 2], device=device)
    association_loss = association_cross_entropy(
        association["augmented_logits"], source_target, target_source
    )

    spatial_geometry = torch.randn(3, config.spatial_geometry_dim, device=device)
    spatial_logits = model.predict_spatial(
        encoded["fused_object"][:3],
        encoded["fused_object"][1:4],
        encoded["fused_parts"][:3],
        encoded["fused_parts"][1:4],
        spatial_geometry,
    )
    spatial_targets = torch.randint(0, 2, spatial_logits.shape, device=device).float()
    spatial_loss = multilabel_focal_loss(spatial_logits, spatial_targets)

    source_tracklets = torch.randn(3, 5, config.dim, device=device)
    target_tracklets = torch.randn(3, 5, config.dim, device=device)
    source_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.bool,
        device=device,
    )
    target_mask = torch.ones(3, 5, dtype=torch.bool, device=device)
    temporal_geometry = torch.randn(3, config.temporal_geometry_dim, device=device)
    temporal_logits = model.predict_temporal(
        source_tracklets,
        target_tracklets,
        source_mask,
        target_mask,
        encoded["fused_parts"][:3],
        encoded["fused_parts"][1:4],
        temporal_geometry,
    )
    temporal_targets = torch.randint(0, 2, temporal_logits.shape, device=device).float()
    temporal_loss = multilabel_focal_loss(temporal_logits, temporal_targets)
    temporal_context = model.temporal(source_tracklets, source_mask)
    consistency_loss = temporal_consistency(temporal_context, source_mask)
    node_logits = model.predict_nodes(encoded["fused_object"])
    node_loss = torch.nn.functional.cross_entropy(
        node_logits, torch.tensor([0, 1, 2, 3], device=device)
    )

    total = (
        rep["diversity"]
        + rep["part_alignment"]
        + rep["object_alignment"]
        + association_loss
        + consistency_loss
        + node_loss
        + spatial_loss
        + temporal_loss
    )
    total.backward()

    pointnet_report = None
    if checkpoint is not None:
        pointnet, pointnet_report = load_geometry_pointnet(checkpoint, device=device)
        with torch.no_grad():
            tokens, anchor = pointnet(torch.randn(2, 6, 128, device=device))
        if tokens.shape != (2, 128, 512) or anchor.shape != (2, 512):
            raise AssertionError("Transferred PointNet returned invalid token shapes.")

    report = {
        "passed": True,
        "device": str(device),
        "model_schema": config.schema_version,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "spatial_predicates": list(SPATIAL_PREDICATES),
        "temporal_predicates": list(TEMPORAL_PREDICATES),
        "object_shape": list(encoded["fused_object"].shape),
        "part_shape": list(encoded["fused_parts"].shape),
        "association_shape_with_dustbin": list(association["augmented_logits"].shape),
        "spatial_logits_shape": list(spatial_logits.shape),
        "temporal_logits_shape": list(temporal_logits.shape),
        "total_smoke_loss": float(total.detach().cpu()),
        "pointnet_transfer": pointnet_report,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end tensor smoke test for IBP-K360.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pointnet-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ibp_model/outputs/smoke_report.json"))
    args = parser.parse_args()
    report = run_smoke(args.device, args.pointnet_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved smoke report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
