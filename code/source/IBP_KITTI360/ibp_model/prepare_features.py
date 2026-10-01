from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from .pointnet import load_geometry_pointnet
from .protocol import OBJECT_CLASSES


SCHEMA_VERSION = "IBP-K360-sensor-features-v1.2.0"
CATEGORY_TO_ID = {name: index for index, name in enumerate(OBJECT_CLASSES)}
OBJECT_CLASSES_JSON = json.dumps(list(OBJECT_CLASSES), separators=(",", ":"))


def stream_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Stream a JSON array using only the standard library."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        started = False
        finished = False
        while not finished:
            chunk = stream.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer.strip():
                break

            cursor = 0
            if not started:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    buffer = ""
                    continue
                if buffer[cursor] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                cursor += 1
                started = True

            while True:
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor < len(buffer) and buffer[cursor] == "]":
                    finished = True
                    cursor += 1
                    break
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise TypeError("Every annotation entry must be a JSON object.")
                yield value
                cursor = end

            buffer = buffer[cursor:]
            if not chunk and not finished:
                raise ValueError(f"Incomplete JSON array: {path}")


def grouped_annotations(path: Path) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[str, int] | None = None
    current: list[dict[str, Any]] = []
    for record in stream_json_array(path):
        key = (str(record["sequence"]), int(record["raw_frame_index"]))
        if current_key is not None and key != current_key:
            yield current
            current = []
        current_key = key
        current.append(record)
    if current:
        yield current


def stable_rng(token: str) -> np.random.Generator:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    return (transform @ homogeneous.T).T[:, :3]


def estimate_normals(points: np.ndarray, neighbours: int = 12) -> np.ndarray:
    if len(points) < 3:
        return np.zeros_like(points, dtype=np.float32)
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
    k = min(neighbours, len(points))
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    normals = np.zeros_like(points, dtype=np.float32)
    for index, neighbours_index in enumerate(indices):
        local = points[neighbours_index]
        local = local - local.mean(axis=0, keepdims=True)
        covariance = local.T @ local / max(len(local) - 1, 1)
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        if np.dot(normal, points[index]) < 0:
            normal = -normal
        normals[index] = normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-8)


def crop_object_points(lidar_xyz: np.ndarray, annotation: dict[str, Any]) -> np.ndarray:
    object_to_sensor = np.asarray(annotation["box_transform_ego"], dtype=np.float32).reshape(4, 4)
    sensor_to_object = np.linalg.inv(object_to_sensor)
    local = transform_points(sensor_to_object, lidar_xyz)
    lower = np.asarray(annotation["box_local_min"], dtype=np.float32) - 1e-3
    upper = np.asarray(annotation["box_local_max"], dtype=np.float32) + 1e-3
    inside = np.logical_and(local >= lower, local <= upper).all(axis=1)
    return lidar_xyz[inside]


def pointnet_input(points: np.ndarray, token: str, count: int = 128) -> np.ndarray | None:
    if len(points) < 10:
        return None
    rng = stable_rng(token)
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    xyz = points[indices].astype(np.float32)
    xyz -= xyz.mean(axis=0, keepdims=True)
    normals = estimate_normals(xyz)
    return np.concatenate([xyz, normals], axis=1).T.astype(np.float32)


def object_crop(
    image: Image.Image,
    instance_mask: np.ndarray | None,
    annotation: dict[str, Any],
) -> tuple[Image.Image | None, str | None]:
    bbox = annotation.get("projected_bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, None
    if not np.isfinite(np.asarray(bbox, dtype=np.float64)).all():
        return None, None
    camera_depth = annotation.get("camera_depth_m")
    if camera_depth is not None and float(camera_depth) <= 0.0:
        return None, None
    width, height = image.size
    x1 = max(0, min(width - 1, int(np.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(np.floor(bbox[1]))))
    x2 = max(0, min(width, int(np.ceil(bbox[2]))))
    y2 = max(0, min(height, int(np.ceil(bbox[3]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, None
    crop = np.asarray(image.crop((x1, y1, x2, y2))).copy()
    crop_source = "projected_3d_box"
    if instance_mask is not None:
        target = int(annotation["combined_instance_id"])
        local_mask = instance_mask[y1:y2, x1:x2] == target
        if local_mask.any():
            crop[~local_mask] = 0
            crop_source = "matched_instance_mask"
    return Image.fromarray(crop), crop_source


class RGBTextEncoders:
    def __init__(
        self,
        clip_model: str,
        caption_model: str,
        device: torch.device,
    ):
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
            CLIPImageProcessor,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            CLIPVisionModel,
        )

        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model)
        self.vision = CLIPVisionModel.from_pretrained(clip_model).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.text = CLIPTextModelWithProjection.from_pretrained(clip_model).to(device).eval()
        self.caption_processor = BlipProcessor.from_pretrained(caption_model)
        self.caption = BlipForConditionalGeneration.from_pretrained(caption_model).to(device).eval()
        for module in (self.vision, self.text, self.caption):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        vision_inputs = self.image_processor(images=images, return_tensors="pt")
        vision_inputs = {key: value.to(self.device) for key, value in vision_inputs.items()}
        patch_tokens = self.vision(**vision_inputs).last_hidden_state[:, 1:]
        if patch_tokens.shape[1:] != (196, 768):
            raise ValueError(
                "The frozen architecture requires CLIP ViT-B/16 patch tokens [196,768]; "
                f"received {tuple(patch_tokens.shape[1:])}."
            )

        caption_inputs = self.caption_processor(images=images, return_tensors="pt")
        caption_inputs = {key: value.to(self.device) for key, value in caption_inputs.items()}
        generated = self.caption.generate(**caption_inputs, max_new_tokens=24)
        captions = self.caption_processor.batch_decode(generated, skip_special_tokens=True)
        text_inputs = self.tokenizer(
            captions, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_embeddings = self.text(**text_inputs).text_embeds
        if text_embeddings.shape[-1] != 512:
            raise ValueError("CLIP projected text embeddings must be 512-dimensional.")
        return patch_tokens.cpu(), text_embeddings.cpu(), captions


class FeatureWriter:
    def __init__(self, path: Path, metadata_path: Path, force: bool):
        if force:
            path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "a")
        self.metadata_path = metadata_path
        self.datasets = self._datasets()
        self._recover_last_commit()
        self.metadata = metadata_path.open("a", encoding="utf-8", newline="\n")

    def _recover_last_commit(self) -> None:
        lengths = [len(dataset) for dataset in self.datasets.values()]
        minimum_rows = min(lengths, default=0)
        committed = int(self.file.attrs.get("completed_row_count", minimum_rows))
        committed = min(committed, minimum_rows)
        for dataset in self.datasets.values():
            if len(dataset) != committed:
                dataset.resize(committed, axis=0)
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as stream:
                lines = [line for line in stream if line.strip()][:committed]
            with self.metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.writelines(lines)
        self.file.attrs["completed_row_count"] = committed
        self.file.flush()

    def _dataset(self, name: str, shape: tuple[int, ...], dtype: str):
        if name in self.file:
            return self.file[name]
        chunks = (1, *shape)
        return self.file.create_dataset(
            name,
            shape=(0, *shape),
            maxshape=(None, *shape),
            chunks=chunks,
            dtype=dtype,
            compression="lzf",
        )

    def _datasets(self) -> dict[str, h5py.Dataset]:
        return {
            "rgb_tokens": self._dataset("rgb_tokens", (196, 768), "float16"),
            "lidar_tokens": self._dataset("lidar_tokens", (128, 512), "float16"),
            "lidar_anchor": self._dataset("lidar_anchor", (512,), "float16"),
            "text_embedding": self._dataset("text_embedding", (512,), "float16"),
            "modality_mask": self._dataset("modality_mask", (3,), "uint8"),
            "category_id": self._dataset("category_id", (), "int16"),
            "center_world": self._dataset("center_world", (3,), "float32"),
            "center_sensor": self._dataset("center_sensor", (3,), "float32"),
            "box_size": self._dataset("box_size", (3,), "float32"),
        }

    @property
    def rows(self) -> int:
        return len(self.datasets["category_id"])

    def append(self, tensors: dict[str, np.ndarray], metadata: dict[str, Any]) -> int:
        row = self.rows
        for name, dataset in self.datasets.items():
            dataset.resize(row + 1, axis=0)
            dataset[row] = tensors[name]
        self.metadata.write(json.dumps({"feature_row": row, **metadata}, separators=(",", ":")) + "\n")
        return row

    def checkpoint(self, raw_frame_index: int) -> None:
        self.file.attrs["last_raw_frame_index"] = raw_frame_index
        self.file.attrs["completed_row_count"] = self.rows
        self.file.attrs["schema_version"] = SCHEMA_VERSION
        self.file.flush()
        self.metadata.flush()

    def close(self) -> None:
        self.metadata.close()
        self.file.close()


def prepare_features(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    annotations_path = args.annotations.resolve()
    output = args.output.resolve()
    metadata_path = output.with_suffix(".jsonl")
    device = torch.device(args.device)

    pointnet = None
    pointnet_metadata = None
    if not args.skip_lidar:
        pointnet, pointnet_metadata = load_geometry_pointnet(
            args.pointnet_checkpoint, device=device, freeze=True
        )
    rgb_text = None
    if not args.skip_rgb_text:
        rgb_text = RGBTextEncoders(args.clip_model, args.caption_model, device)

    writer = FeatureWriter(output, metadata_path, args.force)
    stored_schema = str(writer.file.attrs.get("schema_version", ""))
    stored_classes = str(writer.file.attrs.get("object_classes_json", ""))
    if writer.rows and (
        stored_schema != SCHEMA_VERSION or stored_classes != OBJECT_CLASSES_JSON
    ):
        writer.close()
        raise RuntimeError(
            "Existing feature rows use a stale schema or object vocabulary. "
            "Re-run with --force so category IDs cannot be mixed."
        )
    writer.file.attrs["schema_version"] = SCHEMA_VERSION
    writer.file.attrs["object_classes_json"] = OBJECT_CLASSES_JSON
    writer.file.attrs["num_object_classes"] = len(OBJECT_CLASSES)
    writer.file.flush()
    resume_after = int(writer.file.attrs.get("last_raw_frame_index", -1))
    counters = {
        "frames": 0,
        "objects": writer.rows,
        "rgb_available": int(np.asarray(writer.datasets["modality_mask"][:, 0]).sum()),
        "lidar_available": int(np.asarray(writer.datasets["modality_mask"][:, 1]).sum()),
        "text_available": int(np.asarray(writer.datasets["modality_mask"][:, 2]).sum()),
    }
    captions_path = output.with_suffix(".captions.jsonl")
    captions = captions_path.open("a" if not args.force else "w", encoding="utf-8", newline="\n")
    try:
        for frame_annotations in grouped_annotations(annotations_path):
            raw_frame = int(frame_annotations[0]["raw_frame_index"])
            if args.start_raw_frame is not None and raw_frame < args.start_raw_frame:
                continue
            if args.end_raw_frame is not None and raw_frame > args.end_raw_frame:
                break
            if raw_frame <= resume_after:
                continue
            if args.max_objects is not None and counters["objects"] >= args.max_objects:
                break
            counters["frames"] += 1

            image = None
            instance_mask = None
            image_path = dataset_root / frame_annotations[0]["cam0_filename"]
            mask_name = frame_annotations[0].get("instance_mask_filename")
            mask_path = dataset_root / mask_name if mask_name else None
            if rgb_text is not None and image_path.is_file():
                image = Image.open(image_path).convert("RGB")
                if mask_path is not None and mask_path.is_file():
                    instance_mask = np.asarray(Image.open(mask_path))

            lidar_xyz = None
            lidar_name = frame_annotations[0].get("lidar_filename")
            lidar_path = dataset_root / lidar_name if lidar_name else None
            if pointnet is not None and lidar_path is not None and lidar_path.is_file():
                lidar_xyz = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]

            selected = frame_annotations
            if args.max_objects is not None:
                remaining = args.max_objects - counters["objects"]
                selected = selected[:remaining]

            crops: list[Image.Image] = []
            crop_indices: list[int] = []
            crop_sources: list[str] = []
            if image is not None:
                for index, annotation in enumerate(selected):
                    crop, crop_source = object_crop(image, instance_mask, annotation)
                    if crop is not None:
                        crops.append(crop)
                        crop_indices.append(index)
                        crop_sources.append(str(crop_source))
            rgb_outputs: dict[int, tuple[np.ndarray, np.ndarray, str, str]] = {}
            if crops and rgb_text is not None:
                for start in range(0, len(crops), args.rgb_batch_size):
                    stop = start + args.rgb_batch_size
                    patch, text, batch_captions = rgb_text.encode(crops[start:stop])
                    for local, annotation_index in enumerate(crop_indices[start:stop]):
                        rgb_outputs[annotation_index] = (
                            patch[local].numpy(),
                            text[local].numpy(),
                            batch_captions[local],
                            crop_sources[start + local],
                        )

            lidar_inputs: list[np.ndarray] = []
            lidar_indices: list[int] = []
            if lidar_xyz is not None:
                for index, annotation in enumerate(selected):
                    object_points = crop_object_points(lidar_xyz, annotation)
                    prepared = pointnet_input(object_points, str(annotation["token"]))
                    if prepared is not None:
                        lidar_inputs.append(prepared)
                        lidar_indices.append(index)
            lidar_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if lidar_inputs and pointnet is not None:
                tensor = torch.from_numpy(np.stack(lidar_inputs)).to(device)
                with torch.inference_mode():
                    local_tokens, anchors = pointnet(tensor)
                for local, annotation_index in enumerate(lidar_indices):
                    lidar_outputs[annotation_index] = (
                        local_tokens[local].cpu().numpy(), anchors[local].cpu().numpy()
                    )

            for index, annotation in enumerate(selected):
                rgb_value = rgb_outputs.get(index)
                lidar_value = lidar_outputs.get(index)
                rgb_tokens = np.zeros((196, 768), dtype=np.float16)
                text_embedding = np.zeros(512, dtype=np.float16)
                lidar_tokens = np.zeros((128, 512), dtype=np.float16)
                lidar_anchor = np.zeros(512, dtype=np.float16)
                modality_mask = np.zeros(3, dtype=np.uint8)
                if rgb_value is not None:
                    rgb_tokens = rgb_value[0].astype(np.float16)
                    text_embedding = rgb_value[1].astype(np.float16)
                    modality_mask[[0, 2]] = 1
                    captions.write(
                        json.dumps(
                            {"annotation_token": annotation["token"], "caption": rgb_value[2]},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                if lidar_value is not None:
                    lidar_tokens = lidar_value[0].astype(np.float16)
                    lidar_anchor = lidar_value[1].astype(np.float16)
                    modality_mask[1] = 1

                label = str(annotation["raw_label"])
                if label not in CATEGORY_TO_ID:
                    raise ValueError(f"Unknown KITTI-360 object category: {label}")
                writer.append(
                    {
                        "rgb_tokens": rgb_tokens,
                        "lidar_tokens": lidar_tokens,
                        "lidar_anchor": lidar_anchor,
                        "text_embedding": text_embedding,
                        "modality_mask": modality_mask,
                        "category_id": np.int16(CATEGORY_TO_ID[label]),
                        "center_world": np.asarray(annotation["center_world"], dtype=np.float32),
                        "center_sensor": np.asarray(annotation["center_ego"], dtype=np.float32),
                        "box_size": np.asarray(annotation["box_axis_lengths_m"], dtype=np.float32),
                    },
                    {
                        "annotation_token": annotation["token"],
                        "sample_token": annotation["sample_token"],
                        "scene_token": annotation["scene_token"],
                        "sequence": annotation["sequence"],
                        "raw_frame_index": raw_frame,
                        "instance_token_supervision_only": annotation["instance_token"],
                        "category_name_supervision_only": label,
                        "rgb_crop_source": None if rgb_value is None else rgb_value[3],
                    },
                )
                counters["objects"] += 1
                counters["rgb_available"] += int(modality_mask[0])
                counters["lidar_available"] += int(modality_mask[1])
                counters["text_available"] += int(modality_mask[2])
            writer.checkpoint(raw_frame)
            if counters["frames"] % 25 == 0:
                print(
                    f"Processed {counters['frames']} frames / {counters['objects']} objects; "
                    f"last raw frame {raw_frame}",
                    flush=True,
                )
    finally:
        captions.close()
        writer.close()

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "annotations": str(annotations_path),
        "feature_file": str(output),
        "metadata_file": str(metadata_path),
        "clip_model": None if args.skip_rgb_text else args.clip_model,
        "caption_model": None if args.skip_rgb_text else args.caption_model,
        "pointnet_transfer": pointnet_metadata,
        "counts": counters,
        "object_classes": list(OBJECT_CLASSES),
        "num_object_classes": len(OBJECT_CLASSES),
        "rgb_crop_policy": (
            "Use the projected 3D bounding box; apply the 2D instance mask only when its "
            "encoded ID exactly matches the persistent 3D annotation ID."
        ),
        "privacy_contract": (
            "Persistent instance identity and category name are retained only in metadata for "
            "supervision and evaluation; neither is an encoder input."
        ),
    }
    report_path = output.with_suffix(".report.json")
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen KITTI-360 IBP sensor features.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pointnet-checkpoint", type=Path)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--caption-model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--rgb-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--start-raw-frame", type=int)
    parser.add_argument("--end-raw-frame", type=int)
    parser.add_argument("--skip-rgb-text", action="store_true")
    parser.add_argument("--skip-lidar", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.skip_lidar and args.pointnet_checkpoint is None:
        parser.error("--pointnet-checkpoint is required unless --skip-lidar is used.")
    return args


def main() -> None:
    report = prepare_features(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
