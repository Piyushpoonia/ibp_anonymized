from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


OBJECT_INPUT_KEYS = (
    "rgb_tokens",
    "lidar_tokens",
    "lidar_anchor",
    "text_embedding",
    "modality_mask",
)


def decode_token(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


class ShardPair:
    def __init__(self, feature_path: str | Path, relation_path: str | Path):
        self.feature_path = Path(feature_path).resolve()
        self.relation_path = Path(relation_path).resolve()
        if not self.feature_path.is_file() or not self.relation_path.is_file():
            raise FileNotFoundError(f"Missing feature/relation shard: {self.feature_path}, {self.relation_path}")
        self.features: h5py.File | None = None
        self.relations: h5py.File | None = None

    def open(self) -> tuple[h5py.File, h5py.File]:
        if self.features is None:
            self.features = h5py.File(self.feature_path, "r", swmr=True)
        if self.relations is None:
            self.relations = h5py.File(self.relation_path, "r", swmr=True)
        return self.features, self.relations

    def object(self, row: int) -> dict[str, torch.Tensor]:
        features, _ = self.open()
        return {
            "rgb_tokens": torch.from_numpy(features["rgb_tokens"][row]).float(),
            "lidar_tokens": torch.from_numpy(features["lidar_tokens"][row]).float(),
            "lidar_anchor": torch.from_numpy(features["lidar_anchor"][row]).float(),
            "text_embedding": torch.from_numpy(features["text_embedding"][row]).float(),
            "modality_mask": torch.from_numpy(features["modality_mask"][row]).bool(),
            "category_id": torch.tensor(int(features["category_id"][row]), dtype=torch.long),
        }

    def close(self) -> None:
        if self.features is not None:
            self.features.close()
            self.features = None
        if self.relations is not None:
            self.relations.close()
            self.relations = None


class IndexedShardDataset(Dataset):
    group_name: str

    def __init__(self, feature_paths: list[Path], relation_paths: list[Path]):
        if len(feature_paths) != len(relation_paths) or not feature_paths:
            raise ValueError("Feature and relation shard lists must be non-empty and aligned.")
        self.shards = [ShardPair(feature, relation) for feature, relation in zip(feature_paths, relation_paths)]
        self.offsets = []
        self.lengths = []
        total = 0
        for shard in self.shards:
            with h5py.File(shard.relation_path, "r") as relation_file:
                length = len(relation_file[self.group_name]["targets"])
            self.offsets.append(total)
            self.lengths.append(length)
            total += length
        self.total = total

    def __len__(self) -> int:
        return self.total

    def locate(self, index: int) -> tuple[ShardPair, int]:
        if index < 0:
            index += self.total
        for shard_index in range(len(self.shards) - 1, -1, -1):
            if index >= self.offsets[shard_index]:
                return self.shards[shard_index], index - self.offsets[shard_index]
        raise IndexError(index)

    def close(self) -> None:
        for shard in self.shards:
            shard.close()

    def __del__(self):
        self.close()


class SpatialRelationDataset(IndexedShardDataset):
    group_name = "spatial"

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, row = self.locate(index)
        _, relations = shard.open()
        group = relations[self.group_name]
        result = {
            "source": shard.object(int(group["source_rows"][row])),
            "target": shard.object(int(group["target_rows"][row])),
            "geometry": torch.from_numpy(group["geometry"][row]).float(),
            "labels": torch.from_numpy(group["targets"][row]).float(),
        }
        if "relation_ids" in group:
            result["relation_id"] = decode_token(group["relation_ids"][row])
            result["group_id"] = decode_token(group["group_ids"][row])
            result["sequence"] = decode_token(relations.attrs.get("sequence", ""))
        return result


def stack_tracklet(shard: ShardPair, rows: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
    valid_objects = [shard.object(int(row)) if valid else None for row, valid in zip(rows, mask)]
    template = next((value for value in valid_objects if value is not None), None)
    if template is None:
        raise ValueError("A relation tracklet has no available feature rows.")
    result: dict[str, torch.Tensor] = {}
    for key, template_value in template.items():
        if key == "category_id":
            continue
        values = [
            value[key] if value is not None else torch.zeros_like(template_value)
            for value in valid_objects
        ]
        result[key] = torch.stack(values)
    result["valid_mask"] = torch.from_numpy(mask.astype(np.bool_))
    return result


class TemporalRelationDataset(IndexedShardDataset):
    group_name = "temporal"

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, row = self.locate(index)
        _, relations = shard.open()
        group = relations[self.group_name]
        source_rows = group["source_rows"][row]
        target_rows = group["target_rows"][row]
        source_mask = group["source_mask"][row].astype(bool)
        target_mask = group["target_mask"][row].astype(bool)
        result = {
            "source": stack_tracklet(shard, source_rows, source_mask),
            "target": stack_tracklet(shard, target_rows, target_mask),
            "geometry": torch.from_numpy(group["geometry"][row]).float(),
            "labels": torch.from_numpy(group["targets"][row]).float(),
        }
        if "relation_ids" in group:
            result["relation_id"] = decode_token(group["relation_ids"][row])
            result["group_id"] = decode_token(group["group_ids"][row])
            result["sequence"] = decode_token(relations.attrs.get("sequence", ""))
        return result


class AssociationDataset(Dataset):
    def __init__(self, feature_paths: list[Path], relation_paths: list[Path]):
        if len(feature_paths) != len(relation_paths) or not feature_paths:
            raise ValueError("Feature and relation shard lists must be non-empty and aligned.")
        self.shards = [ShardPair(feature, relation) for feature, relation in zip(feature_paths, relation_paths)]
        self.offsets = []
        self.lengths = []
        total = 0
        for shard in self.shards:
            with h5py.File(shard.relation_path, "r") as relation_file:
                length = len(relation_file["association"]["shape"])
            self.offsets.append(total)
            self.lengths.append(length)
            total += length
        self.total = total

    def __len__(self) -> int:
        return self.total

    def _locate(self, index: int) -> tuple[ShardPair, int]:
        for shard_index in range(len(self.shards) - 1, -1, -1):
            if index >= self.offsets[shard_index]:
                return self.shards[shard_index], index - self.offsets[shard_index]
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, row = self._locate(index)
        _, relations = shard.open()
        group = relations["association"]
        ns, nt = (int(value) for value in group["shape"][row])
        source_rows = np.asarray(group["source_rows"][row], dtype=np.int64)
        target_rows = np.asarray(group["target_rows"][row], dtype=np.int64)

        def object_set(rows: np.ndarray) -> dict[str, torch.Tensor]:
            objects = [shard.object(int(feature_row)) for feature_row in rows]
            return {
                key: torch.stack([obj[key] for obj in objects])
                for key in objects[0]
            }

        return {
            "source": object_set(source_rows),
            "target": object_set(target_rows),
            "geometry": torch.from_numpy(
                np.asarray(group["geometry"][row], dtype=np.float32).reshape(ns, nt, 16)
            ),
            "candidate_mask": torch.from_numpy(
                np.asarray(group["candidate_mask"][row], dtype=np.uint8).reshape(ns, nt).astype(bool)
            ),
            "source_targets": torch.from_numpy(
                np.asarray(group["source_targets"][row], dtype=np.int64)
            ),
            "target_targets": torch.from_numpy(
                np.asarray(group["target_targets"][row], dtype=np.int64)
            ),
        }

    def close(self) -> None:
        for shard in self.shards:
            shard.close()

    def __del__(self):
        self.close()
