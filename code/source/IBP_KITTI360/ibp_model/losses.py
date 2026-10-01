from __future__ import annotations

import torch
import torch.nn.functional as F


def component_diversity(
    parts: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    if valid is not None:
        parts = parts[valid]
    if parts.numel() == 0:
        return parts.sum() * 0.0
    normalized = F.normalize(parts, dim=-1)
    similarity = normalized @ normalized.transpose(-2, -1)
    eye = torch.eye(parts.size(-2), dtype=torch.bool, device=parts.device)
    return similarity.masked_select(~eye.unsqueeze(0)).square().mean()


def symmetric_info_nce(
    first: torch.Tensor,
    second: torch.Tensor,
    valid: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    if valid is not None:
        first = first[valid]
        second = second[valid]
    if first.size(0) < 2:
        return first.sum() * 0.0
    first = F.normalize(first.reshape(-1, first.size(-1)), dim=-1)
    second = F.normalize(second.reshape(-1, second.size(-1)), dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def representation_losses(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = outputs["modality_mask"]
    both = mask[:, 0] & mask[:, 1]
    rgb_text = mask[:, 0] & mask[:, 2]
    lidar_text = mask[:, 1] & mask[:, 2]
    if outputs["rgb_parts"].size(-2) > 1:
        diversity = component_diversity(
            outputs["rgb_parts"], mask[:, 0]
        ) + component_diversity(outputs["lidar_parts"], mask[:, 1])
        part = symmetric_info_nce(outputs["rgb_parts"], outputs["lidar_parts"], both)
    else:
        zero = outputs["fused_object"].sum() * 0.0
        diversity = zero
        part = zero
    object_alignment = 0.5 * (
        symmetric_info_nce(outputs["rgb_object"], outputs["text_object"], rgb_text)
        + symmetric_info_nce(outputs["lidar_object"], outputs["text_object"], lidar_text)
    )
    return {
        "diversity": diversity,
        "part_alignment": part,
        "object_alignment": object_alignment,
    }


def association_cross_entropy(
    augmented_logits: torch.Tensor,
    source_target: torch.Tensor,
    target_source: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional CE; each target index may point to the opposite dustbin."""
    row = F.cross_entropy(augmented_logits[:-1], source_target.long())
    column = F.cross_entropy(augmented_logits[:, :-1].T, target_source.long())
    return 0.5 * (row + column)


def temporal_consistency(context: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    losses = []
    normalized = F.normalize(context, dim=-1)
    for index in range(normalized.size(0)):
        values = normalized[index][valid_mask[index]]
        if values.size(0) > 1:
            losses.append((1.0 - (values[:-1] * values[1:]).sum(dim=-1)).mean())
    return torch.stack(losses).mean() if losses else context.sum() * 0.0


def multilabel_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    probability = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), probability, 1.0 - probability)
    return (((1.0 - pt) ** gamma) * bce).mean()


def symmetric_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first = torch.sigmoid(first_logits).clamp(1e-6, 1.0 - 1e-6)
    second = torch.sigmoid(second_logits).clamp(1e-6, 1.0 - 1e-6)
    first_dist = torch.stack([first, 1.0 - first], dim=-1)
    second_dist = torch.stack([second, 1.0 - second], dim=-1)
    return 0.5 * (
        F.kl_div(first_dist.log(), second_dist, reduction="batchmean")
        + F.kl_div(second_dist.log(), first_dist, reduction="batchmean")
    )
