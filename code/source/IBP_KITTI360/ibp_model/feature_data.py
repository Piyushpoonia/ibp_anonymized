from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_KEYS = (
    "rgb_tokens",
    "lidar_tokens",
    "lidar_anchor",
    "text_embedding",
    "modality_mask",
    "category_id",
    "center_world",
    "center_sensor",
    "box_size",
)


class FeatureShardDataset(Dataset):
    """Lazy reader for one or more sequence-level HDF5 feature shards."""

    def __init__(self, paths: list[str | Path], require_all_modalities: bool = False):
        self.paths = [Path(path).resolve() for path in paths]
        self.require_all_modalities = require_all_modalities
        if not self.paths:
            raise ValueError("At least one feature shard is required.")
        self.lengths: list[int] = []
        self.offsets: list[int] = []
        self.row_indices: list[np.ndarray] = []
        total = 0
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Feature shard not found: {path}")
            with h5py.File(path, "r") as feature_file:
                missing = [key for key in FEATURE_KEYS if key not in feature_file]
                if missing:
                    raise ValueError(f"Feature shard {path} is missing datasets: {missing}")
                raw_length = len(feature_file["category_id"])
                if any(len(feature_file[key]) != raw_length for key in FEATURE_KEYS):
                    raise ValueError(f"Feature shard {path} contains inconsistent row counts.")
                if require_all_modalities:
                    modality_mask = np.asarray(feature_file["modality_mask"], dtype=np.bool_)
                    rows = np.flatnonzero(modality_mask.all(axis=1)).astype(np.int64)
                else:
                    rows = np.arange(raw_length, dtype=np.int64)
                length = len(rows)
            self.offsets.append(total)
            self.lengths.append(length)
            self.row_indices.append(rows)
            total += length
        self.total = total
        if self.total == 0:
            protocol = "full-multimodal" if require_all_modalities else "unfiltered"
            raise RuntimeError(f"No {protocol} object features were found in the requested shards.")
        self._files: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return self.total

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        for shard in range(len(self.paths) - 1, -1, -1):
            if index >= self.offsets[shard]:
                local_index = index - self.offsets[shard]
                return shard, int(self.row_indices[shard][local_index])
        raise IndexError(index)

    def _file(self, shard: int) -> h5py.File:
        if shard not in self._files:
            self._files[shard] = h5py.File(self.paths[shard], "r", swmr=True)
        return self._files[shard]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard, row = self._locate(index)
        feature_file = self._file(shard)
        return {
            "rgb_tokens": torch.from_numpy(feature_file["rgb_tokens"][row]).float(),
            "lidar_tokens": torch.from_numpy(feature_file["lidar_tokens"][row]).float(),
            "lidar_anchor": torch.from_numpy(feature_file["lidar_anchor"][row]).float(),
            "text_embedding": torch.from_numpy(feature_file["text_embedding"][row]).float(),
            "modality_mask": torch.from_numpy(feature_file["modality_mask"][row]).bool(),
            "category_id": torch.tensor(int(feature_file["category_id"][row]), dtype=torch.long),
            "center_world": torch.from_numpy(feature_file["center_world"][row]).float(),
            "center_sensor": torch.from_numpy(feature_file["center_sensor"][row]).float(),
            "box_size": torch.from_numpy(feature_file["box_size"][row]).float(),
        }

    def close(self) -> None:
        for feature_file in self._files.values():
            feature_file.close()
        self._files.clear()

    def __del__(self):
        self.close()
