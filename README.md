# INTEGRATION-BY-PARTS-COMPOSITIONAL-MULTI--MODAL-LEARNING-FOR-DYNAMIC-3D-SCENE-GRAPH-GENERATION

# IBP-K360 reproduction package

This package contains the KITTI-360 preprocessing pipeline, the three-stage
IBP model, the frozen feature-extraction weights, and a reference test result.
The raw KITTI-360 dataset is **not** included. Place a licensed copy in
`KITTI-360/` at the package root, or pass its location with `--dataset-root`.

## Contents

| Path | Purpose |
| --- | --- |
| `code/data_preparation/` | Timeline construction, spatial and temporal predicate generation, association targets, feature and relation shards. |
| `code/source/` | Model modules, Stage 1/2/3 training, predicted tracklets, evaluation, and the frozen recovery rule. |
| `resource/` | Frozen PointNet checkpoint and the CLIP ViT-B/16 and BLIP base cache archive in three parts. |
| `result/` | An unchanged seed-42, held-out 0007+0009 JSON result using validation-selected tracklet recovery. |
| `tests/` | Packaging and launcher checks. |

The fixed split is train drives 0000, 0002, 0003, 0004, 0005, and 0010;
validation drive 0006; held-out test drives 0007 and 0009. The split and
predicate list are in
`code/source/IBP_KITTI360/splits/kitti360_ibp_sequence_split_v1.json`.

## Requirements

- Linux, Python 3.12, a compatible NVIDIA CUDA driver and a GPU with enough
  memory for feature extraction and training. CPU-only execution is not the
  reproduced configuration.
- A licensed KITTI-360 raw dataset with the listed drives, calibration,
  rectified RGB, 3D boxes, 2D semantics, poses, and Velodyne scans. The
  preparer checks for these inputs before starting.
- Enough disk space for the raw data, the generated feature/relations dataset
  (approximately 64 GiB in the source experiment), checkpoints, and caches.
  Space and runtime vary by system.

From the package root, create an environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch 2.6.0 build appropriate for your driver first.
python -m pip install -r requirements_model.txt -r requirements_predicates.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python resource/install_weights.py
```

The bundled archive has the two model caches used by preprocessing:
`openai/clip-vit-base-patch16` and
`Salesforce/blip-image-captioning-base`. The PointNet `obj_enc.pth` checkpoint
is already in `resource/`. The installer checks the original archive SHA-256
before extraction and runs entirely from local files. It writes the expanded
cache to `resource/hf_home/`, which is ignored by Git.

## Raw dataset layout

Use the official KITTI-360 directory layout. The root must contain, among
other files:

```text
KITTI-360/
  calibration/perspective.txt
  calibration/calib_cam_to_pose.txt
  calibration/calib_cam_to_velo.txt
  data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/
  data_2d_semantics/train/2013_05_28_drive_0007_sync/image_00/semantic/
  data_3d_raw/2013_05_28_drive_0007_sync/velodyne_points/data/
  data_3d_bboxes/train/2013_05_28_drive_0007_sync.xml
  data_poses/2013_05_28_drive_0007_sync/
```

The equivalent paths must exist for all nine split sequences. Do not rename
the official drive directories. This package never uploads the raw dataset.

## Full run

Run these commands from the package root after activating the environment:

```bash
# Build timelines, predicates, features, association candidates and HDF5 shards.
python code/data_preparation/prepare.py --step all --device cuda

# Train Stage 1 (40 epochs), Stage 2 (30 epochs), Stage 3 (10 epochs),
# generate predicted tracklets, and evaluate on held-out drives 0007 and 0009.
python code/source/train.py --phase all --seed 42 --device cuda
```

`prepare.py` generates spatial predicates (`left_of`, `in_front_of`, `near`,
`overlapping`, `occluding`) and temporal predicates (`approaching`,
`moving_away`, `same_motion_direction`) from the raw scene timelines. It then
encodes CLIP RGB/text, BLIP captions and PointNet LiDAR features and builds
the relation and association files used by training. Generated data goes to
`PREPARED_DATASET/` by default. Stages and outputs go to `RUNS/seed_42/`.

The default training sequence is:

1. Stage 1 learns the object/component representation.
2. Stage 2 warms up association and relation heads with Stage 1 weights.
3. Predicted tracklets are generated from Stage 2 for train and validation.
4. Stage 3 trains the relation model using predicted tracklets.
5. Predicted test tracklets are generated; the base test result is saved under
   `RUNS/seed_42/final_test/final_test_metrics.json`.
6. The fixed recovery rule selected on validation drive 0006 is applied to
   test tracklets; its distinct result is saved under
   `RUNS/seed_42/final_test_recovered/final_test_metrics.json`.

The recovery selection file is
`code/source/IBP_KITTI360/config/frozen_recovery_selection.json`; it is not
retuned on test data. Existing completed stages and shards are preserved on
rerun. For an individual step use `--step timeline|predicates|features|relations`
or `--phase stage1|stage2|tracks|stage3|evaluate|recover`. The `--force` option
on preparation regenerates outputs; do not use it when resuming a partial run.
The exact commands can be inspected without data or a GPU with `--dry-run`.
Use `--dataset-root`, `--prepared-root`, and `--output-root` to relocate large
files if necessary.

## Step-by-step run

Run each step from the package root on a Linux machine with a CUDA GPU. Wait
for one command to finish successfully before starting the next. If you cloned
this package from GitHub, run `git lfs pull` first so the weight parts are real
files rather than Git LFS pointers.

### 1. Set up the environment and raw data

Place the official raw dataset at `final_submission/KITTI-360/`, preserving
the directory layout shown above. Then:

```bash
cd /path/to/final_submission
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch 2.6.0 build for your system first.
python -m pip install -r requirements_model.txt -r requirements_predicates.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python resource/install_weights.py
```

The last command verifies and unpacks the CLIP and BLIP weights. PointNet is
already present as `resource/obj_enc.pth`.

### 2. Build the timeline

```bash
python code/data_preparation/prepare.py --step timeline
```

This reads the raw RGB, LiDAR, boxes, poses, and calibration and creates
`PREPARED_DATASET/timeline/parts/` for all nine drives. It also checks that
the required raw directories are present.

### 3. Generate and validate predicates

```bash
python code/data_preparation/prepare.py --step predicates
```

This writes spatial and temporal predicate labels to
`PREPARED_DATASET/predicates/parts/` and validates each drive's labels.

### 4. Extract object features

```bash
python code/data_preparation/prepare.py --step features --device cuda
```

This applies the frozen CLIP, BLIP, and PointNet encoders and writes one
feature HDF5 file per drive under `PREPARED_DATASET/features/`. This step can
take substantial time; leave the raw dataset accessible throughout.

### 5. Build association targets and relation files

```bash
python code/data_preparation/prepare.py --step relations
```

This creates association candidates and the HDF5 relation indices under
`PREPARED_DATASET/relations/`. Stages 1-3 require these files.

### 6. Train Stage 1

```bash
python code/source/train.py --phase stage1 --seed 42 --device cuda
```

The 40-epoch component/object encoder run writes
`RUNS/seed_42/stage1/checkpoints/stage1_best_macro_f1.pt`.

### 7. Train Stage 2

```bash
python code/source/train.py --phase stage2 --seed 42 --device cuda
```

The 30-epoch warm-up uses the Stage 1 checkpoint and writes
`RUNS/seed_42/stage2_warmup/checkpoints/stage2_warmup_best.pt`.

### 8. Generate training and validation tracklets

```bash
python code/source/train.py --phase tracks --seed 42 --device cuda
```

This uses the Stage 2 model to write predicted tracklet shards for the six
training drives and validation drive 0006 to `RUNS/seed_42/predicted_tracklets/`.

### 9. Train Stage 3

```bash
python code/source/train.py --phase stage3 --seed 42 --device cuda
```

The 10-epoch predicted-tracklet run writes
`RUNS/seed_42/stage3_predicted/checkpoints/stage3_best.pt`. Stage 3 also
generates any missing training or validation tracklet shards automatically.

### 10. Evaluate the base model on held-out drives

```bash
python code/source/train.py --phase evaluate --seed 42 --device cuda
```

This generates predicted tracklets for test drives 0007 and 0009 and writes
the base result to `RUNS/seed_42/final_test/final_test_metrics.json`.

### 11. Evaluate with validation-selected tracklet recovery

```bash
python code/source/train.py --phase recover --seed 42 --device cuda
```

This applies the fixed rule selected on validation drive 0006, then writes
`RUNS/seed_42/final_test_recovered/final_test_metrics.json`. This is the
setting of the bundled JSON in `result/`; it is distinct from Step 10.

Commands skip completed outputs, so rerun a step after an interruption. Do
not pass `--force` unless you deliberately intend to regenerate prepared data.
To inspect the command sequence without raw data or a GPU, add `--dry-run` to
either launcher. If the raw or generated data is elsewhere, pass the same
`--dataset-root` or `--output-root` to every preparation step, and pass its
location with `--prepared-root` to every training step.

## Result provenance

`result/seed42_final_test_recovered.json` is copied without modifying its
content. It reports the **recovered-tracklet** seed-42 held-out 0007+0009
result, not the base no-recovery result. Its checkpoint and recovery metadata
remain in the JSON. Reported values include spatial macro-F1 93.76%, spatial
mAP 98.31%, temporal macro-F1 85.57%, temporal mAP 86.27%, association
accuracy 98.82%, and predicted-tracklet coverage 84.36%. The JSON is a
reference artifact, not a substitute for rerunning the pipeline.

The package was assembled and its launchers and source tests are checked
locally, but a complete raw-data-to-results rerun has **not** been performed
from this packaged directory. CUDA availability, software versions, data
licenses, and checkpoint provenance can affect exact reproducibility.

## Local checks

```bash
python -m unittest discover -s tests -v
PYTHONPATH=code/source python -m unittest discover \
  -s code/source/IBP_KITTI360/tests -v
python code/data_preparation/prepare.py --dry-run
python code/source/train.py --dry-run
```

## Publishing to GitHub

The model archive is split into three roughly 1 GiB files, each tracked by
Git LFS through `.gitattributes`. Install Git LFS **before adding files**:

```bash
git lfs install
git init
git add .
git lfs ls-files
git status
```

Do not commit raw KITTI-360 data, generated shards, expanded caches, or
checkpoints; `.gitignore` excludes these directories. Check the upstream
license and redistribution terms for each pretrained weight before making a
public repository. Git LFS storage/bandwidth quotas may apply.
