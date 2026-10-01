from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


GEOMETRY_CHANNELS_FROM_OCRL = (0, 1, 2, 6, 7, 8)


class SpatialTransformer3D(nn.Module):
    def __init__(self, channels: int = 6, out_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.fc1 = nn.Linear(out_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch = points.size(0)
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2).values
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).reshape(batch, 3, 3)
        identity = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
        return x + identity


class GeometryPointNet(nn.Module):
    """OCRL PointNet transferred from XYZ+RGB+normal to XYZ+normal input."""

    def __init__(self, channels: int = 6, out_dim: int = 512):
        super().__init__()
        self.stn = SpatialTransformer3D(channels=channels, out_dim=out_dim)
        self.conv1 = nn.Conv1d(channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.size(1) != 6:
            raise ValueError(f"Expected PointNet input [B, 6, N], got {tuple(points.shape)}")
        transform = self.stn(points)
        xyz = torch.bmm(points[:, :3].transpose(2, 1), transform).transpose(2, 1)
        features = torch.cat([xyz, points[:, 3:]], dim=1)
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        tokens = self.bn3(self.conv3(x)).transpose(1, 2)
        return tokens, tokens.max(dim=1).values


def load_geometry_pointnet(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> tuple[GeometryPointNet, dict[str, object]]:
    """Load all compatible OCRL weights and remove only the three RGB channels."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OCRL PointNet checkpoint not found: {checkpoint}")
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise TypeError("The OCRL PointNet checkpoint must contain a state dictionary.")

    model = GeometryPointNet()
    target = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    sliced = []
    for name, value in source.items():
        if name not in target:
            continue
        if name in {"stn.conv1.weight", "conv1.weight"}:
            value = value[:, GEOMETRY_CHANNELS_FROM_OCRL, :].contiguous()
            sliced.append(name)
        if target[name].shape != value.shape:
            raise ValueError(
                f"Incompatible OCRL tensor {name}: expected {tuple(target[name].shape)}, "
                f"found {tuple(value.shape)}"
            )
        transferred[name] = value

    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if missing or unexpected:
        raise ValueError(f"Incomplete PointNet transfer; missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_channels": "XYZ+RGB+normal",
        "target_channels": "XYZ+normal",
        "selected_source_channel_indices": list(GEOMETRY_CHANNELS_FROM_OCRL),
        "sliced_tensors": sliced,
        "transferred_tensors": len(transferred),
        "frozen": freeze,
    }
    return model, metadata
