# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A computer vision research project that fine-tunes [GeoCLIP](https://github.com/ViktorooReps/geoclip) to predict GPS coordinates from photos taken in Italy. All notebooks are designed to run on **Google Colab** (GPU required) with **Google Drive** as persistent storage. The local `.venv` (Python 3.13) can be used for lightweight exploration but not for training.

## Environment

**Local**: `.venv` at project root (Python 3.13). Activate with `source .venv/bin/activate`. Only used for editing; not suitable for GPU training.

**Colab**: Run notebooks on Google Colab. Install deps each session with the install cell at the top of each notebook.

**Key packages**: `geoclip`, `torch`, `torchvision`, `pandas`, `scikit-learn`, `tqdm`, `peft`, `folium`, `seaborn`, `beautifulsoup4`, `Pillow`, `geopy`, `datasets`, `global-land-mask`, `scipy`

## Notebook Structure

Each notebook is self-contained and designed to run top-to-bottom. Run cells in order; section headers describe what can be skipped.

| Notebook | Purpose |
|----------|---------|
| `GeoClip_Training_V1.ipynb` | Standard fine-tuning on GLDv2 Italy subset; frozen CLIP + last 2 ViT layers unfrozen; `checkpoints_v1` |
| `GeoClip_Training_V2.ipynb` | Adds PEFT/LoRA (rank=8, alpha=16) on q\_proj+v\_proj; same GLDv2 dataset; `checkpoints_v2` |
| `GeoClip_Training_V3.ipynb` | Switches dataset to **OSV5M Italy** (111K train / 1.4K test); same LoRA arch; `checkpoints_v3`; no land penalty |
| `GeoClip_Training_V3_1.ipynb` | **Current training notebook**: adds land-aware penalty loss; `global_land_mask` + `scipy` distance transform; `checkpoints_v3_1` |
| `GeoClip_Evaluation_V1.ipynb` | Evaluates `checkpoints_v1` model on GLDv2 test split |
| `GeoClip_Evaluation_V2.ipynb` | Evaluates `checkpoints_v2` model on GLDv2 test split |
| `GeoClip_Evaluation_V3.ipynb` | **Current evaluation notebook**: evaluates `checkpoints_v3_1` model on OSV5M fixed test split (1,397 images) |
| `osv5m/osv5m_dataset_preprocess.ipynb` | Preprocesses OSV5M Italy dataset: filters by bounding box, splits by region subdirs and 20-bucket tar archives |
| `scraper/Google_Landmarks_Download_and_GPS_Scraper.ipynb` | Legacy pipeline: downloads GLDv2 shards from S3, scrapes GPS, filters for Italy |
| `old/Vision_Project.ipynb` | Initial baseline: GeoCLIP on Google Landmarks, GPS scraping from Wikimedia |

## Architecture

### GeoClipItaly (current — V3/V3.1)
- **Backbone**: OpenAI CLIP ViT-Large-Patch14 (via `geoclip.model.GeoCLIP`)
- **Frozen layers**: all CLIP params; LoRA adapters injected on `q_proj` + `v_proj` in every ViT attention layer
- **LoRA config**: rank=8, alpha=16, dropout=0.05, `task_type=FEATURE_EXTRACTION`
- **Unfrozen non-LoRA**: `visual_projection` and `image_encoder.mlp` (domain adaptation)
- **Regression head**: `512 → 1024 → 512 → 256 → 2` with BatchNorm1d, GELU, Dropout(0.3/0.2/0.1), final Sigmoid
- **Trainable params**: ~3.74M / 440M total (0.85%)
- **Output**: normalized (lat, lon) ∈ [0, 1], mapped back via `denorm()` to Italy bounding box

> **Note**: the V1 architecture used a different head (`512 → 1024 → 512 → 128 → 2`, Dropout 0.2 only) and unfroze the last 2 ViT encoder layers directly instead of using LoRA.

### Italy bounding box
- Lat: `[35.4, 47.2]` (training and scraper filter)
- Lon: `[6.6, 18.8]` (training and scraper filter)

### `denorm(norm_lat, norm_lon)`
```python
lat = norm_lat * (MAX_LAT - MIN_LAT) + MIN_LAT
lon = norm_lon * (MAX_LON - MIN_LON) + MIN_LON
```

## Dataset

### Current: OSV5M Italy (V3 / V3.1)
- **Source**: [OSV5M dataset](https://github.com/gastruc/osv5m), filtered to Italy bounding box
- **Preprocessing notebook**: `osv5m/osv5m_dataset_preprocess.ipynb`
- **Train+val pool**: ~111,787 images (split 85/15 into train/val at runtime, `random_state=42`)
- **Fixed test set**: 1,397 images (from OSV5M's own test split — not derived from train metadata)
- **Image storage**: 20-bucket subdirectory scheme (`image_id % 20` → `00..19`), one uncompressed TAR per bucket per split, cached on Drive
- **Region distribution (train pool)**: Nord 55,022 / Sud\_Isole 22,065 / Centro 18,455

### Legacy: Google Landmarks Dataset v2 (V1 / V2)
- ~35,827 cleaned Italy images after CLIP fingerprint dedup
- Stored as `dataset_coordinate_FINGERPRINT.csv` + `dataset_images.zip`
- Test split derived via `train_test_split(test_size=0.15, random_state=42)`

## Loss Function

### V3.1 (current): Combined loss with land penalty
```python
LAND_BUFFER_KM      = 25.0   # grace zone for coastal/boat shots
LAND_SCALE_KM       = 150.0  # distance at which penalty reaches 1.0
LAND_PENALTY_WEIGHT = 0.20

def combined_loss(pred, target):
    base = 0.05 * l1_loss(pred, target) + 0.95 * haversine_loss(pred, target)
    return base + LAND_PENALTY_WEIGHT * land_penalty_loss(pred)
```
- `land_penalty_loss`: builds a 300×300 penalty grid over Italy using `global_land_mask` + `scipy.ndimage.distance_transform_edt`; samples it differentiably via `F.grid_sample` (bilinear). Zero on land or within 25 km of coast; ramps to 1.0 at 150 km offshore.

### V3 (no land penalty)
```python
def combined_loss(pred, target):
    return 0.05 * l1_loss(pred, target) + 0.95 * haversine_loss(pred, target)
```

### V1 / V2: same formula as V3

## Google Drive Layout

```
/content/drive/MyDrive/Vision_Project_2026/
├── osv5m/
│   └── osv5m_italy_dataset/
│       ├── train_images/
│       │   ├── train_metadata.csv          # ~111,787 rows (id, latitude, longitude)
│       │   ├── 00/ .. 19/                  # bucket subdirs with .jpg files
│       │   └── (train_subdirtars cached by deploy cell)
│       └── test_images/
│           ├── test_metadata.csv           # 1,397 rows
│           ├── 00/ .. 19/
│           └── (test_subdirtars cached by deploy cell)
└── checkpoints_v3_1/
    ├── geoclip_italy_BEST.pth              # best checkpoint (lowest val loss)
    ├── checkpoint_epoch_005.pth            # periodic full checkpoints
    └── training_curves.png
```

**Legacy GLDv2 paths** (V1/V2):
```
/content/drive/MyDrive/Unipd/Computer Vision Project/
├── Dataset_GeoClip_CLEANED/
│   ├── dataset_coordinate_FINGERPRINT.csv
│   └── dataset_images.zip
└── Dataset_GeoClip_V5/  (V1)  /  Dataset_GeoClip_V6/ (V2)
    └── geoclip_italy_BEST.pth
```

## Training Configuration (V3.1)

| Hyperparameter | Value |
|---|---|
| Batch size | 32 (T4) |
| Val/test batch | 128 (4× multiplier) |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Total epochs | 15 |
| Early stop patience | 5 |
| Checkpoint frequency | every 5 epochs |
| AMP | enabled |
| Grad clip | 1.0 |
| LR schedule | CosineAnnealingLR, eta\_min=1e-6 |
| Val split | 15%, random\_state=42 |

## Version History & Key Results

| Version | Dataset | Checkpoint dir | Trainable strategy | Best val loss | Best checkpoint error |
|---|---|---|---|---|---|
| V1 | GLDv2 Italy (~35K) | checkpoints\_v1 | Last 2 ViT layers + head | 0.16167 | 165.2 km |
| V2 | GLDv2 Italy (~35K) | checkpoints\_v2 | LoRA rank=8 + head | 0.11837 | 121.0 km |
| V3 | OSV5M Italy (111K) | checkpoints\_v3 | LoRA rank=8 + head | — | — |
| V3.1 | OSV5M Italy (111K) | checkpoints\_v3\_1 | LoRA rank=8 + head + land penalty | 0.11579 (ep 4) | 118.3 km |
| V3.2 | OSV5M Italy (111K) | checkpoints\_v3\_2 | LoRA rank=16, all 4 attn projs + head | — | — |

### Evaluation results on fixed OSV5M test set (1,397 images) — V3.1 epoch 4 checkpoint

| Model | Mean error | Median error | Within 200 km | Within 500 km |
|---|---|---|---|---|
| Baseline GeoCLIP | 1290.5 km | 538.3 km | 24.6% | 47.2% |
| GeoClipItaly (V3.1, ep 4) | 195.2 km | 124.2 km | 66.1% | 91.6% |

### Evaluation results on GLDv2 test split — for reference

| Version | Model | Mean error | Median error | Within 200 km |
|---|---|---|---|---|
| V1 | Baseline | 617.1 km | 216.0 km | 48.5% |
| V1 | Fine-tuned | 177.3 km | 168.0 km | 63.6% |
| V2 | Baseline | 1015.8 km | 244.7 km | — |
| V2 | Fine-tuned | 157.0 km | 103.3 km | — |

> Baseline numbers differ between eval runs because the baseline uses `model.predict()` (top-1 GPS gallery lookup) which varies with geoclip version; the fine-tuned model uses direct regression.

## Image Preprocessing (must match training exactly)

```python
# Training (with augmentation)
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# Val / test / inference (no augmentation)
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])
```

## Evaluation Metrics

- **Primary**: Haversine distance (km) between predicted and true GPS
- **Accuracy thresholds**: % of images within 1 / 25 / 50 / 100 / 200 / 500 km
- **Macro region classification**: Nord (lat > 44.0), Centro (lat > 41.5), Sud/Isole
- **MC Dropout**: 50 forward passes with dropout enabled to produce uncertainty heatmaps

## Common Pitfalls

- **Transformers version**: newer versions return `BaseModelOutputWithPooling` instead of a tensor. Both V3.1 training and V3 evaluation notebooks access `vm_base = vision_model.base_model` and extract `pooler_output` (with fallback to `last_hidden_state[:, 0, :]`). The evaluation notebook also patches `baseline_model.image_encoder.forward` for the same reason.
- **MC Dropout**: call `enable_dropout(model)` after `model.eval()` to activate only dropout layers. BatchNorm stays frozen because `GeoClipItaly.train()` always calls `self.geoclip.eval()`.
- **`model.train()` override**: `GeoClipItaly.train()` forces `self.geoclip.eval()` to prevent BatchNorm/Dropout in the frozen CLIP backbone from changing behavior.
- **LoRA forward pass**: use `vision_model.base_model(pixel_values=x)` not `vision_model(pixel_values=x)` to avoid PeftModel's generic forward conflicting with the `pixel_values` kwarg.
- **Checkpoint compatibility**: loading a pre-LoRA checkpoint into a LoRA model will report missing LoRA keys — this is expected. The optimizer/scheduler state is skipped in that case and starts fresh.
- **Land penalty grid**: built once at setup time (~5 s) using `global_land_mask` and `scipy`. Must be re-built if `MIN_LAT/MAX_LAT/MIN_LON/MAX_LON` change.
- **OSV5M deploy**: `deploy_split()` is idempotent — already-extracted buckets are skipped by comparing file counts. Safe to re-run each Colab session.
- **Test split reproducibility**: OSV5M version uses the fixed `test_metadata.csv` (not a random split). GLDv2 versions used `random_state=42, test_size=0.15`.
- **`torchao` conflict**: the install cell uninstalls `torchao` before installing packages to avoid version conflicts with the Colab default.
