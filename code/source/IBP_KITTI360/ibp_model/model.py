from __future__ import annotations

import math

import torch
import torch.nn as nn

from .protocol import ModelConfig, validate_protocol


def _make_transformer(dim: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class SharedPartEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.dim
        self.use_parts = config.use_parts
        self.rgb_projection = nn.Linear(config.rgb_input_dim, d)
        self.lidar_projection = nn.Linear(config.lidar_input_dim, d)
        self.text_projection = nn.Linear(config.text_input_dim, d)
        self.anchor_projection = nn.Linear(config.anchor_input_dim, d)
        self.rgb_position = nn.Parameter(torch.randn(1, 196, d) * 0.01)
        self.lidar_position = nn.Parameter(torch.randn(1, 128, d) * 0.01)
        if self.use_parts:
            self.queries = nn.Parameter(torch.randn(config.num_parts, d) * 0.02)
            self.rgb_attention = nn.MultiheadAttention(
                d, config.num_heads, config.dropout, batch_first=True
            )
            self.lidar_attention = nn.MultiheadAttention(
                d, config.num_heads, config.dropout, batch_first=True
            )
            self.part_pool = nn.Linear(d, 1)
        self.missing_rgb = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.missing_lidar = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.missing_text = nn.Parameter(torch.randn(1, d) * 0.02)
        self.missing_anchor = nn.Parameter(torch.randn(1, d) * 0.02)
        self.fusion_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.global_fusion = _make_transformer(
            d, config.num_heads, config.fusion_layers, config.dropout
        )
        if self.use_parts:
            self.part_fusion = _make_transformer(d, config.num_heads, 1, config.dropout)
            self.global_part_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(d)
        self.register_buffer(
            "enabled_modality_mask",
            torch.tensor(
                [name in config.enabled_modalities for name in ("rgb", "lidar", "text")],
                dtype=torch.bool,
            ),
            persistent=False,
        )

    @staticmethod
    def _availability(mask: torch.Tensor, column: int) -> torch.Tensor:
        return mask[:, column].bool()

    def _tokens_or_missing(
        self,
        tokens: torch.Tensor,
        available: torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor:
        fallback = missing.expand(tokens.size(0), tokens.size(1), -1)
        return torch.where(available[:, None, None], tokens, fallback)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        lidar_anchor: torch.Tensor,
        text_embedding: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if modality_mask.shape[-1] != 3:
            raise ValueError("modality_mask columns must be [RGB, LiDAR, text].")
        effective_mask = modality_mask.bool() & self.enabled_modality_mask.unsqueeze(0)
        rgb_available = self._availability(effective_mask, 0)
        lidar_available = self._availability(effective_mask, 1)
        text_available = self._availability(effective_mask, 2)

        rgb = self.rgb_projection(rgb_tokens)
        lidar = self.lidar_projection(lidar_tokens)
        rgb = rgb + self.rgb_position[:, : rgb.size(1)]
        lidar = lidar + self.lidar_position[:, : lidar.size(1)]
        rgb = self._tokens_or_missing(rgb, rgb_available, self.missing_rgb)
        lidar = self._tokens_or_missing(lidar, lidar_available, self.missing_lidar)

        if self.use_parts:
            queries = self.queries.unsqueeze(0).expand(rgb.size(0), -1, -1)
            rgb_parts, rgb_attention = self.rgb_attention(
                queries, rgb, rgb, need_weights=True, average_attn_weights=False
            )
            lidar_parts, lidar_attention = self.lidar_attention(
                queries, lidar, lidar, need_weights=True, average_attn_weights=False
            )
            rgb_weights = torch.softmax(self.part_pool(rgb_parts), dim=1)
            lidar_weights = torch.softmax(self.part_pool(lidar_parts), dim=1)
            rgb_object = (rgb_weights * rgb_parts).sum(dim=1)
            lidar_object = (lidar_weights * lidar_parts).sum(dim=1)
        else:
            rgb_object = rgb.mean(dim=1)
            lidar_object = lidar.mean(dim=1)

        text = self.text_projection(text_embedding)
        anchor = self.anchor_projection(lidar_anchor)
        text = torch.where(text_available[:, None], text, self.missing_text.expand_as(text))
        anchor = torch.where(lidar_available[:, None], anchor, self.missing_anchor.expand_as(anchor))

        fusion = self.fusion_token.expand(rgb.size(0), -1, -1)
        global_sequence = torch.cat(
            [fusion, rgb_object[:, None], lidar_object[:, None], anchor[:, None], text[:, None]], dim=1
        )
        global_padding = torch.stack(
            [
                torch.zeros_like(rgb_available),
                ~rgb_available,
                ~lidar_available,
                ~lidar_available,
                ~text_available,
            ],
            dim=1,
        )
        global_object = self.global_fusion(
            global_sequence, src_key_padding_mask=global_padding
        )[:, 0]

        if self.use_parts:
            both = rgb_available & lidar_available
            fused_parts = torch.where(
                both[:, None, None],
                0.5 * (rgb_parts + lidar_parts),
                torch.where(rgb_available[:, None, None], rgb_parts, lidar_parts),
            )
            fused_parts = self.part_fusion(fused_parts)
            part_object = (
                torch.softmax(self.part_pool(fused_parts), dim=1) * fused_parts
            ).sum(dim=1)
            gate = self.global_part_gate(torch.cat([global_object, part_object], dim=-1))
            fused_object = self.output_norm(global_object + gate * part_object)
        else:
            fused_object = self.output_norm(global_object)
            part_object = torch.zeros_like(global_object)
            fused_parts = fused_object[:, None]
            rgb_parts = rgb_object[:, None]
            lidar_parts = lidar_object[:, None]
            rgb_attention = rgb.new_zeros((rgb.size(0), 0, rgb.size(1)))
            lidar_attention = lidar.new_zeros((lidar.size(0), 0, lidar.size(1)))
            gate = torch.zeros_like(global_object)
        return {
            "fused_object": fused_object,
            "global_object": global_object,
            "part_object": part_object,
            "rgb_object": rgb_object,
            "lidar_object": lidar_object,
            "text_object": text,
            "rgb_parts": rgb_parts,
            "lidar_parts": lidar_parts,
            "fused_parts": fused_parts,
            "rgb_attention": rgb_attention,
            "lidar_attention": lidar_attention,
            "global_part_gate": gate,
            "modality_mask": effective_mask,
        }


class GeometryAwareAssociation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.dim
        self.temperature = config.sinkhorn_temperature
        self.iterations = config.sinkhorn_iterations
        self.score = nn.Sequential(
            nn.LayerNorm(d * 4 + config.association_geometry_dim),
            nn.Linear(d * 4 + config.association_geometry_dim, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, 1),
        )
        self.source_dustbin = nn.Parameter(torch.tensor(0.0))
        self.target_dustbin = nn.Parameter(torch.tensor(0.0))
        self.dustbin_corner = nn.Parameter(torch.tensor(0.0))

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        log_assignment = logits / self.temperature
        for _ in range(self.iterations):
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=1, keepdim=True)
            log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=0, keepdim=True)
        return log_assignment.exp()

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_geometry: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ns, nt = source.size(0), target.size(0)
        if pair_geometry.shape != (ns, nt, self.score[0].normalized_shape[0] - source.size(-1) * 4):
            raise ValueError("pair_geometry must have shape [source_objects, target_objects, G].")
        src = source[:, None, :].expand(ns, nt, -1)
        dst = target[None, :, :].expand(ns, nt, -1)
        pair = torch.cat([src, dst, src - dst, src * dst, pair_geometry], dim=-1)
        scores = self.score(pair).squeeze(-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -1e4)

        augmented = scores.new_empty((ns + 1, nt + 1))
        augmented[:ns, :nt] = scores
        augmented[:ns, nt] = self.source_dustbin
        augmented[ns, :nt] = self.target_dustbin
        augmented[ns, nt] = self.dustbin_corner
        return {
            "pair_logits": scores,
            "augmented_logits": augmented,
            "soft_assignment": self._sinkhorn(augmented),
        }


class TrackletTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.position = nn.Parameter(
            torch.randn(1, config.tracklet_length, config.dim) * 0.01
        )
        self.missing = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.encoder = _make_transformer(
            config.dim, config.num_heads, config.temporal_layers, config.dropout
        )
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, tracklets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tracklets.ndim != 3 or valid_mask.shape != tracklets.shape[:2]:
            raise ValueError("tracklets must be [B,T,D] and valid_mask must be [B,T].")
        if (~valid_mask.bool()).all(dim=1).any():
            raise ValueError("Every tracklet must contain at least one valid observation.")
        missing = self.missing.expand(tracklets.size(0), tracklets.size(1), -1)
        x = torch.where(valid_mask[:, :, None], tracklets, missing)
        x = x + self.position[:, : x.size(1)]
        return self.norm(self.encoder(x, src_key_padding_mask=~valid_mask.bool()))


class PartPairInteraction(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.dim * 4),
            nn.Linear(config.dim * 4, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )

    def forward(self, source_parts: torch.Tensor, target_parts: torch.Tensor) -> torch.Tensor:
        source = source_parts.mean(dim=-2)
        target = target_parts.mean(dim=-2)
        return self.net(torch.cat([source, target, source - target, source * target], dim=-1))


class PairHead(nn.Module):
    def __init__(self, config: ModelConfig, geometry_dim: int, outputs: int):
        super().__init__()
        width = config.dim * 5 + geometry_dim
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim * 2, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, outputs),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        part_interaction: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.cat(
            [source, target, source - target, source * target, part_interaction, geometry], dim=-1
        )
        return self.net(pair)


class IBPK360Model(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        validate_protocol(self.config)
        self.object_encoder = SharedPartEncoder(self.config)
        self.association = GeometryAwareAssociation(self.config)
        self.temporal = TrackletTransformer(self.config)
        self.part_interaction = (
            PartPairInteraction(self.config) if self.config.use_parts else None
        )
        self.node_head = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dim, self.config.num_object_classes),
        )
        self.spatial_head = PairHead(
            self.config,
            self.config.spatial_geometry_dim,
            self.config.num_spatial_predicates,
        )
        self.temporal_head = PairHead(
            self.config,
            self.config.temporal_geometry_dim,
            self.config.num_temporal_predicates,
        )

    def encode_objects(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.object_encoder(**inputs)

    def predict_nodes(self, contextual_objects: torch.Tensor) -> torch.Tensor:
        return self.node_head(contextual_objects)

    def predict_spatial(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_parts: torch.Tensor,
        target_parts: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        interaction = (
            self.part_interaction(source_parts, target_parts)
            if self.part_interaction is not None
            else torch.zeros_like(source)
        )
        return self.spatial_head(source, target, interaction, geometry)

    def predict_temporal(
        self,
        source_tracklets: torch.Tensor,
        target_tracklets: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        source_parts: torch.Tensor,
        target_parts: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        source_context = self.temporal(source_tracklets, source_mask)
        target_context = self.temporal(target_tracklets, target_mask)
        middle = source_tracklets.size(1) // 2
        interaction = (
            self.part_interaction(source_parts, target_parts)
            if self.part_interaction is not None
            else torch.zeros_like(source_context[:, middle])
        )
        return self.temporal_head(
            source_context[:, middle], target_context[:, middle], interaction, geometry
        )
