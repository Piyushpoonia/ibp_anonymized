#!/usr/bin/env python3
"""Build a portable interactive viewer for consecutive KITTI-360 predicates."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


SPATIAL_PREDICATES = [
    "left_of",
    "in_front_of",
    "near",
    "overlapping",
    "occluding",
]
TEMPORAL_PREDICATES = [
    "approaching",
    "moving_away",
    "same_motion_direction",
    "crossing_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--part-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--raw-stride", type=int, default=5)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without loading the complete file."""
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

            position = 0
            if not started:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    buffer = ""
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                started = True
                position += 1

            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    buffer = ""
                    break
                if buffer[position] == "]":
                    finished = True
                    buffer = buffer[position + 1 :]
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    buffer = buffer[position:]
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object records in {path}")
                yield value
                position = end


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def positive_labels(record: dict[str, Any], allowed: list[str]) -> list[str]:
    values = record.get("positive_predicates", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in allowed]


def scan_spatial_support(path: Path) -> tuple[dict[int, int], dict[int, str]]:
    scores: dict[int, int] = defaultdict(int)
    images: dict[int, str] = {}
    for record in iter_json_array(path):
        frame = int(record["raw_frame_index"])
        images.setdefault(frame, str(record["rgb_left_filename"]))
        labels = positive_labels(record, SPATIAL_PREDICATES)
        if labels:
            visible = valid_box(record.get("subject_projected_bbox_xyxy")) is not None
            visible = visible and valid_box(record.get("object_projected_bbox_xyxy")) is not None
            scores[frame] += len(labels) * (2 if visible else 1)
        else:
            scores.setdefault(frame, 0)
    return dict(scores), images


def contiguous_runs(frames: list[int], stride: int) -> list[list[int]]:
    if not frames:
        return []
    runs: list[list[int]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame - runs[-1][-1] == stride:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def select_frames(
    scores: dict[int, int], sample_count: int, stride: int, start_frame: int | None
) -> list[int]:
    if sample_count <= 0:
        raise ValueError("sample-count must be positive")
    frames = sorted(scores)
    runs = contiguous_runs(frames, stride)
    if start_frame is not None:
        for run in runs:
            candidates = [index for index, frame in enumerate(run) if frame >= start_frame]
            if candidates:
                start = candidates[0]
                chosen = run[start : start + sample_count]
                if len(chosen) == sample_count:
                    return chosen
        raise ValueError(
            f"No {sample_count}-sample contiguous run exists at or after frame {start_frame}."
        )

    best: tuple[int, int, list[int]] | None = None
    for run in runs:
        if len(run) < sample_count:
            continue
        window_score = sum(scores[frame] for frame in run[:sample_count])
        candidate = (window_score, -run[0], run[:sample_count])
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        for start in range(1, len(run) - sample_count + 1):
            window_score += scores[run[start + sample_count - 1]]
            window_score -= scores[run[start - 1]]
            chosen = run[start : start + sample_count]
            candidate = (window_score, -chosen[0], chosen)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        longest = max((len(run) for run in runs), default=0)
        raise ValueError(
            f"No contiguous run contains {sample_count} samples at stride {stride}; "
            f"the longest run contains {longest}."
        )
    return best[2]


def object_key(record: dict[str, Any], prefix: str) -> str:
    value = record.get(f"{prefix}_instance_token")
    if value:
        return str(value)
    value = record.get(f"{prefix}_annotation_token")
    return str(value or f"unknown_{prefix}")


def add_object(
    objects: dict[str, dict[str, Any]],
    key: str,
    label: str,
    box: Any,
    instance_id: Any = None,
) -> None:
    candidate = {
        "key": key,
        "label": label,
        "bbox": valid_box(box),
        "instance_id": instance_id,
    }
    current = objects.get(key)
    if current is None or (current["bbox"] is None and candidate["bbox"] is not None):
        objects[key] = candidate


def collect_annotations(
    path: Path, selected: set[int]
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[int, str]]:
    objects: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    images: dict[int, str] = {}
    for record in iter_json_array(path):
        frame = int(record["raw_frame_index"])
        if frame not in selected:
            continue
        key = str(record.get("instance_token") or record["token"])
        add_object(
            objects[frame],
            key,
            str(record.get("raw_label", "object")),
            record.get("projected_bbox_xyxy"),
            record.get("instance_id"),
        )
        images.setdefault(frame, str(record["cam0_filename"]))
    return dict(objects), images


def collect_spatial_relations(
    path: Path,
    selected: set[int],
    objects: dict[int, dict[str, dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    relations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_json_array(path):
        frame = int(record["raw_frame_index"])
        if frame not in selected:
            continue
        labels = positive_labels(record, SPATIAL_PREDICATES)
        if not labels:
            continue
        subject_key = object_key(record, "subject")
        target_key = object_key(record, "object")
        add_object(
            objects[frame], subject_key, str(record.get("subject_label", "object")),
            record.get("subject_projected_bbox_xyxy")
        )
        add_object(
            objects[frame], target_key, str(record.get("object_label", "object")),
            record.get("object_projected_bbox_xyxy")
        )
        relations[frame].append(
            {
                "relation_id": str(record.get("token", "")),
                "subject_key": subject_key,
                "object_key": target_key,
                "subject_label": str(record.get("subject_label", "object")),
                "object_label": str(record.get("object_label", "object")),
                "predicates": labels,
            }
        )
    return dict(relations)


def collect_temporal_relations(
    path: Path,
    selected: set[int],
    objects: dict[int, dict[str, dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    relations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return {}
    for record in iter_json_array(path):
        frame = int(record["middle_raw_frame_index"])
        if frame not in selected:
            continue
        labels = positive_labels(record, TEMPORAL_PREDICATES)
        if not labels:
            continue
        raw_frames = [int(value) for value in record.get("raw_frame_indices", [])]
        try:
            middle = raw_frames.index(frame)
        except ValueError:
            middle = len(raw_frames) // 2
        subject_boxes = record.get("subject_projected_bboxes_xyxy", [])
        object_boxes = record.get("object_projected_bboxes_xyxy", [])
        subject_box = subject_boxes[middle] if middle < len(subject_boxes) else None
        target_box = object_boxes[middle] if middle < len(object_boxes) else None
        subject_key = object_key(record, "subject")
        target_key = object_key(record, "object")
        add_object(
            objects[frame], subject_key, str(record.get("subject_label", "object")), subject_box
        )
        add_object(
            objects[frame], target_key, str(record.get("object_label", "object")), target_box
        )
        relations[frame].append(
            {
                "relation_id": str(record.get("token", "")),
                "subject_key": subject_key,
                "object_key": target_key,
                "subject_label": str(record.get("subject_label", "object")),
                "object_label": str(record.get("object_label", "object")),
                "predicates": labels,
                "radial_velocity_mps": record.get(
                    "subject_radial_velocity_toward_object_mps"
                ),
            }
        )
    return dict(relations)


def export_image(source: Path, target: Path, quality: int) -> tuple[int, int]:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        size = rgb.size
        rgb.save(target, "JPEG", quality=quality, optimize=True)
    return size


def assign_display_ids(objects: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(
        objects.values(),
        key=lambda item: (
            item["bbox"] is None,
            item["bbox"][0] if item["bbox"] else 1e9,
            item["label"],
            item["key"],
        ),
    )
    for index, item in enumerate(ordered, start=1):
        item["display_id"] = f"O{index}"


def viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KITTI-360 Predicate Sequence Viewer</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <div>
      <h1>KITTI-360 Predicate Sequence</h1>
      <p id="sequenceLine"></p>
    </div>
    <div class="summary" id="summaryLine"></div>
  </header>
  <nav class="toolbar" aria-label="Playback controls">
    <button id="previous" class="icon" title="Previous sample" aria-label="Previous sample">&#9664;</button>
    <button id="play" class="icon primary" title="Play" aria-label="Play">&#9654;</button>
    <button id="next" class="icon" title="Next sample" aria-label="Next sample">&#9654;&#124;</button>
    <input id="frameSlider" type="range" min="0" value="0" aria-label="Sample position">
    <strong id="frameCounter"></strong>
    <label>Speed
      <select id="speed">
        <option value="1000">0.5x</option>
        <option value="525" selected>Natural</option>
        <option value="260">2x</option>
        <option value="130">4x</option>
      </select>
    </label>
  </nav>
  <div class="display-options">
    <label><input id="showBoxes" type="checkbox" checked> Object boxes</label>
    <label><input id="showEdges" type="checkbox" checked> Relation edges</label>
    <label><input id="showSpatial" type="checkbox" checked> Spatial predicates</label>
    <label><input id="showTemporal" type="checkbox" checked> Temporal predicates</label>
  </div>
  <main>
    <section class="viewer-band">
      <div id="stage" class="stage">
        <img id="frameImage" alt="KITTI-360 sequence frame">
        <canvas id="overlay"></canvas>
      </div>
      <div class="frame-meta">
        <span id="rawFrame"></span>
        <span id="objectCount"></span>
        <span id="relationCount"></span>
      </div>
      <div class="legend" id="legend"></div>
    </section>
    <aside>
      <h2>Relations in this sample</h2>
      <div id="relationList"></div>
    </aside>
  </main>
  <footer>
    Candidate annotations. Temporal labels use world-frame motion with ego motion removed.
  </footer>
  <script src="viewer_data.js"></script>
  <script src="viewer.js"></script>
</body>
</html>
"""


def viewer_css() -> str:
    return """:root {
  font-family: Arial, Helvetica, sans-serif;
  color: #17202a;
  background: #f4f6f8;
}
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; }
header {
  display: flex; justify-content: space-between; align-items: flex-end; gap: 24px;
  padding: 16px 22px; background: #ffffff; border-bottom: 1px solid #cfd6dd;
}
h1 { margin: 0 0 5px; font-size: 22px; letter-spacing: 0; }
h2 { margin: 0; padding: 14px 16px; font-size: 16px; border-bottom: 1px solid #d8dde3; }
p { margin: 0; color: #53606d; font-size: 13px; }
.summary { font-size: 13px; font-weight: 700; text-align: right; }
.toolbar, .display-options {
  display: flex; align-items: center; gap: 10px; padding: 10px 18px;
  background: #ffffff; border-bottom: 1px solid #d8dde3;
}
.toolbar input[type="range"] { flex: 1; min-width: 120px; accent-color: #1565c0; }
.toolbar label, .display-options label { font-size: 13px; white-space: nowrap; }
.icon {
  width: 34px; height: 34px; border: 1px solid #aeb7c1; background: #ffffff;
  color: #17202a; cursor: pointer; font-size: 15px;
}
.icon:hover { background: #edf3f8; }
.icon.primary { background: #1565c0; border-color: #1565c0; color: #ffffff; }
select { height: 30px; border: 1px solid #aeb7c1; background: #ffffff; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 390px; min-height: 0; }
.viewer-band { padding: 14px 18px 18px; min-width: 0; }
.stage { position: relative; width: 100%; background: #111820; overflow: hidden; }
.stage img { display: block; width: 100%; height: auto; }
.stage canvas { position: absolute; inset: 0; pointer-events: none; }
.frame-meta {
  display: flex; justify-content: space-between; gap: 16px; padding: 9px 2px;
  font-size: 13px; font-weight: 700;
}
.legend { display: flex; flex-wrap: wrap; gap: 7px 12px; font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 14px; height: 4px; display: inline-block; }
aside { background: #ffffff; border-left: 1px solid #cfd6dd; min-width: 0; }
#relationList { max-height: calc(100vh - 205px); overflow-y: auto; }
.relation-group { border-bottom: 1px solid #e1e5ea; padding: 9px 12px; }
.relation-group h3 { margin: 0 0 7px; font-size: 12px; text-transform: uppercase; color: #53606d; }
.relation-row { padding: 7px 4px; border-top: 1px solid #eef1f4; font-size: 12px; line-height: 1.35; }
.relation-row:first-of-type { border-top: 0; }
.relation-row strong { font-size: 12px; }
.tag { display: inline-block; margin: 3px 4px 0 0; padding: 2px 5px; color: white; font-weight: 700; }
.muted { color: #7b8793; }
.empty { padding: 24px 16px; color: #6f7b87; }
footer { padding: 9px 18px; border-top: 1px solid #cfd6dd; font-size: 11px; color: #687480; background: #ffffff; }
@media (max-width: 900px) {
  header { align-items: flex-start; flex-direction: column; }
  .summary { text-align: left; }
  main { grid-template-columns: 1fr; }
  aside { border-left: 0; border-top: 1px solid #cfd6dd; }
  #relationList { max-height: none; }
  .display-options { flex-wrap: wrap; }
}
"""


def viewer_js() -> str:
    return """(() => {
  const data = window.VIEWER_DATA;
  const colors = {
    left_of: '#1565c0', in_front_of: '#00897b', near: '#7b1fa2',
    overlapping: '#ef6c00', occluding: '#c62828', approaching: '#2e7d32',
    moving_away: '#ad1457', same_motion_direction: '#455a64', crossing_path: '#6d4c41'
  };
  const shortName = {
    left_of: 'left', in_front_of: 'front', near: 'near', overlapping: 'overlap',
    occluding: 'occlude', approaching: 'approach', moving_away: 'away',
    same_motion_direction: 'same motion', crossing_path: 'crossing'
  };
  const el = id => document.getElementById(id);
  const image = el('frameImage');
  const canvas = el('overlay');
  const context = canvas.getContext('2d');
  const slider = el('frameSlider');
  let index = 0;
  let timer = null;

  el('sequenceLine').textContent = `${data.sequence} | every ${data.raw_stride}th raw frame`;
  el('summaryLine').textContent = `${data.frames.length} displayed samples | raw ${data.start_frame}-${data.end_frame}`;
  slider.max = Math.max(0, data.frames.length - 1);
  el('legend').innerHTML = data.predicate_order.map(name =>
    `<span><i class="swatch" style="background:${colors[name]}"></i>${name.replaceAll('_', ' ')}</span>`
  ).join('');

  function enabledRelations(frame) {
    const result = [];
    if (el('showSpatial').checked) result.push(...frame.spatial.map(r => ({...r, kind: 'spatial'})));
    if (el('showTemporal').checked) result.push(...frame.temporal.map(r => ({...r, kind: 'temporal'})));
    return result;
  }

  function objectMap(frame) {
    return new Map(frame.objects.map(object => [object.key, object]));
  }

  function center(box, sx, sy) {
    return [((box[0] + box[2]) / 2) * sx, ((box[1] + box[3]) / 2) * sy];
  }

  function drawLabel(text, x, y, color) {
    context.font = 'bold 11px Arial';
    const width = context.measureText(text).width + 8;
    context.fillStyle = 'rgba(255,255,255,0.92)';
    context.fillRect(x - 2, y - 12, width, 16);
    context.fillStyle = color;
    context.fillText(text, x + 2, y);
  }

  function draw() {
    const frame = data.frames[index];
    const cssWidth = image.clientWidth;
    const cssHeight = image.clientHeight;
    if (!cssWidth || !cssHeight) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    const sx = cssWidth / frame.width;
    const sy = cssHeight / frame.height;
    const objects = objectMap(frame);

    if (el('showBoxes').checked) {
      for (const object of frame.objects) {
        if (!object.bbox) continue;
        const [x1, y1, x2, y2] = object.bbox;
        context.strokeStyle = '#00e5ff';
        context.lineWidth = 2;
        context.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        drawLabel(`${object.display_id} ${object.label}`, x1 * sx, Math.max(14, y1 * sy), '#005b70');
      }
    }

    if (el('showEdges').checked) {
      for (const relation of enabledRelations(frame)) {
        const subject = objects.get(relation.subject_key);
        const target = objects.get(relation.object_key);
        if (!subject?.bbox || !target?.bbox) continue;
        const a = center(subject.bbox, sx, sy);
        const b = center(target.bbox, sx, sy);
        relation.predicates.forEach((predicate, offsetIndex) => {
          const color = colors[predicate] || '#263238';
          const dx = b[0] - a[0];
          const dy = b[1] - a[1];
          const length = Math.max(1, Math.hypot(dx, dy));
          const offset = (offsetIndex - (relation.predicates.length - 1) / 2) * 5;
          const ox = -dy / length * offset;
          const oy = dx / length * offset;
          context.strokeStyle = color;
          context.lineWidth = relation.kind === 'temporal' ? 3 : 2;
          context.setLineDash(relation.kind === 'temporal' ? [7, 4] : []);
          context.beginPath();
          context.moveTo(a[0] + ox, a[1] + oy);
          context.lineTo(b[0] + ox, b[1] + oy);
          context.stroke();
          context.setLineDash([]);
          drawLabel(shortName[predicate] || predicate, (a[0] + b[0]) / 2 + ox, (a[1] + b[1]) / 2 + oy, color);
        });
      }
    }
  }

  function relationRows(relations, objects, kind) {
    if (!relations.length) return `<div class="relation-group"><h3>${kind}</h3><div class="muted">No positive ${kind.toLowerCase()} predicate</div></div>`;
    return `<div class="relation-group"><h3>${kind}</h3>${relations.map(relation => {
      const subject = objects.get(relation.subject_key);
      const target = objects.get(relation.object_key);
      const visible = Boolean(subject?.bbox && target?.bbox);
      const tags = relation.predicates.map(name => `<span class="tag" style="background:${colors[name] || '#263238'}">${name.replaceAll('_', ' ')}</span>`).join('');
      const velocity = relation.radial_velocity_mps == null ? '' : `<div class="muted">radial velocity: ${Number(relation.radial_velocity_mps).toFixed(2)} m/s</div>`;
      return `<div class="relation-row"><strong>${subject?.display_id || '?'} ${relation.subject_label} &rarr; ${target?.display_id || '?'} ${relation.object_label}</strong><div>${tags}</div>${velocity}${visible ? '' : '<div class="muted">one or both objects are outside the RGB view</div>'}</div>`;
    }).join('')}</div>`;
  }

  function render() {
    const frame = data.frames[index];
    slider.value = index;
    el('frameCounter').textContent = `${index + 1} / ${data.frames.length}`;
    el('rawFrame').textContent = `Raw frame ${frame.raw_frame}`;
    el('objectCount').textContent = `${frame.objects.length} objects`;
    el('relationCount').textContent = `${frame.spatial.length} spatial + ${frame.temporal.length} temporal pairs`;
    const objects = objectMap(frame);
    el('relationList').innerHTML = relationRows(frame.spatial, objects, 'Spatial') + relationRows(frame.temporal, objects, 'Temporal');
    image.onload = draw;
    image.src = frame.image;
    if (image.complete) draw();
  }

  function stop() {
    if (timer) window.clearInterval(timer);
    timer = null;
    el('play').innerHTML = '&#9654;';
    el('play').title = 'Play';
  }

  function play() {
    stop();
    el('play').innerHTML = '&#10074;&#10074;';
    el('play').title = 'Pause';
    timer = window.setInterval(() => {
      index = (index + 1) % data.frames.length;
      render();
    }, Number(el('speed').value));
  }

  el('previous').addEventListener('click', () => { stop(); index = (index - 1 + data.frames.length) % data.frames.length; render(); });
  el('next').addEventListener('click', () => { stop(); index = (index + 1) % data.frames.length; render(); });
  el('play').addEventListener('click', () => timer ? stop() : play());
  slider.addEventListener('input', () => { stop(); index = Number(slider.value); render(); });
  el('speed').addEventListener('change', () => { if (timer) play(); });
  ['showBoxes', 'showEdges', 'showSpatial', 'showTemporal'].forEach(id => el(id).addEventListener('change', () => { draw(); render(); }));
  window.addEventListener('resize', draw);
  window.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft') el('previous').click();
    if (event.key === 'ArrowRight') el('next').click();
    if (event.key === ' ') { event.preventDefault(); el('play').click(); }
  });
  render();
})();
"""


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    part_root = args.part_root.resolve()
    output_root = args.output_root.resolve()
    spatial_path = part_root / "spatial_relations.json"
    annotation_path = part_root / "sample_annotations.json"
    temporal_path = part_root / "temporal" / "temporal_relations.json"
    for path in (dataset_root, part_root, spatial_path, annotation_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force to replace it: {output_root}")
        shutil.rmtree(output_root)
    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True)

    print("Scanning spatial support and contiguous key samples...", flush=True)
    scores, relation_images = scan_spatial_support(spatial_path)
    selected_frames = select_frames(
        scores, args.sample_count, args.raw_stride, args.start_frame
    )
    selected_set = set(selected_frames)
    print(
        f"Selected {len(selected_frames)} samples from raw frame "
        f"{selected_frames[0]} to {selected_frames[-1]}.",
        flush=True,
    )

    print("Collecting visible objects...", flush=True)
    objects, annotation_images = collect_annotations(annotation_path, selected_set)
    print("Collecting all positive spatial predicates...", flush=True)
    spatial = collect_spatial_relations(spatial_path, selected_set, objects)
    print("Collecting all positive temporal predicates...", flush=True)
    temporal = collect_temporal_relations(temporal_path, selected_set, objects)

    frames: list[dict[str, Any]] = []
    spatial_count = temporal_count = 0
    for index, frame in enumerate(selected_frames, start=1):
        source_name = annotation_images.get(frame) or relation_images.get(frame)
        if not source_name:
            raise ValueError(f"No RGB image is recorded for raw frame {frame}")
        source = dataset_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target_name = f"{index:03d}_raw_{frame:010d}.jpg"
        width, height = export_image(source, frames_root / target_name, args.jpeg_quality)
        frame_objects = objects.get(frame, {})
        assign_display_ids(frame_objects)
        frame_spatial = spatial.get(frame, [])
        frame_temporal = temporal.get(frame, [])
        spatial_count += len(frame_spatial)
        temporal_count += len(frame_temporal)
        frames.append(
            {
                "sample_number": index,
                "raw_frame": frame,
                "image": f"frames/{target_name}",
                "width": width,
                "height": height,
                "objects": list(frame_objects.values()),
                "spatial": frame_spatial,
                "temporal": frame_temporal,
            }
        )
        if index % 10 == 0 or index == len(selected_frames):
            print(f"Exported sample {index}/{len(selected_frames)}", flush=True)

    sequence = part_root.name
    payload = {
        "schema_version": "IBP-K360-continuous-viewer-v1.0.0",
        "sequence": sequence,
        "sample_count": len(frames),
        "raw_stride": args.raw_stride,
        "start_frame": selected_frames[0],
        "end_frame": selected_frames[-1],
        "predicate_order": SPATIAL_PREDICATES + TEMPORAL_PREDICATES,
        "frames": frames,
    }
    (output_root / "index.html").write_text(viewer_html(), encoding="utf-8")
    (output_root / "style.css").write_text(viewer_css(), encoding="utf-8")
    (output_root / "viewer.js").write_text(viewer_js(), encoding="utf-8")
    (output_root / "viewer_data.js").write_text(
        "window.VIEWER_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    report = {
        "sequence": sequence,
        "sample_count": len(frames),
        "raw_stride": args.raw_stride,
        "start_frame": selected_frames[0],
        "end_frame": selected_frames[-1],
        "spatial_positive_pairs_in_viewer": spatial_count,
        "temporal_positive_pairs_in_viewer": temporal_count,
        "temporal_file_present": temporal_path.is_file(),
        "open_file": str(output_root / "index.html"),
    }
    (output_root / "viewer_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Viewer ready: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
