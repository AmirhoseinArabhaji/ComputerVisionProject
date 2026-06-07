# Report of Experiments and Changes Across GeoClip Notebooks

This report summarizes what was changed and what experiment each notebook represents.  
It is based on the notebook code and available cell outputs; I did not re-run the notebooks.

---

## 1. Overall experiment progression

The notebooks show a clear experimental path:

1. **Training V1**: First Italy fine-tuning attempt using the original GLDv2-derived Italy dataset and partial backbone unfreezing.
2. **Training V2**: Same dataset family, but changed the adaptation strategy to LoRA.
3. **Training V3**: Switched from the GLDv2 Italy dataset to the larger OSV5M Italy dataset.
4. **Training V3.1**: Added a land-aware loss penalty to reduce sea/off-land predictions.
5. **Training V3.2**: Increased LoRA capacity and added optimization improvements such as gradient accumulation, warmup, and optional `torch.compile`.
6. **Evaluation V1–V3**: Separate evaluation notebooks that compare baseline GeoCLIP against the corresponding fine-tuned checkpoints.

---

# Training notebooks

## GeoClip_Training_V1.ipynb

### Purpose
This was the first main fine-tuning experiment on the Italy dataset created from `Italy_Dataset_Builder.ipynb`.

### Dataset
- Uses `ItalyDataset/italy_master.csv`.
- Uses three region TAR archives: `nord`, `centro`, and `sud_isole`.
- Images are extracted to `/content/gld_images`.
- Region filter is set to `None`, so all Italian regions are used.
- Available output shows:
  - Found images: **34,183 / 34,183**
  - Missing images: **0**
  - Train / val / test split is created from this dataset.

### Model experiment
- Uses pretrained GeoCLIP as the backbone.
- Freezes most of GeoCLIP.
- Unfreezes the last **2 ViT encoder layers**.
- Also unfreezes:
  - `visual_projection`
  - `image_encoder.mlp`
  - the custom regression head
- Regression head:
  - `512 → 1024 → 512 → 256 → 2`
  - Uses BatchNorm, GELU, Dropout, and final Sigmoid output for normalized latitude/longitude.

### Loss and training
- Loss combines:
  - `0.05 * L1 loss`
  - `0.95 * Haversine loss`
- Batch size: **16**
- Learning rate: **1e-4**
- Weight decay: **1e-4**
- Epochs: **30**
- Checkpoint saved every epoch.
- Uses AMP and gradient clipping.
- Scheduler: cosine annealing.

### Result shown in output
The visible output shows training resumed around epoch 22 and finished at epoch 30.

Best validation loss shown:
- **Best val loss: 0.16167**
- Best validation distance around **165 km**

### Interpretation
V1 tests whether partially unfreezing the original GeoCLIP image backbone is enough for Italy geolocation. It improves domain adaptation but trains more backbone parameters than later LoRA versions.

---

## GeoClip_Training_V2.ipynb

### Purpose
This notebook changes the fine-tuning strategy from direct partial backbone unfreezing to **LoRA-based fine-tuning**.

### Dataset
- Still uses the GLDv2-derived `ItalyDataset`.
- Uses `italy_master.csv` and region TARs.
- Available output shows:
  - Found images: **32,687**
  - Missing images: **32,591**
  - Trainable rows: **32,687**
  - Split:
    - Train: **23,615**
    - Validation: **4,168**
    - Test: **4,904**

### Main change from V1
Instead of unfreezing the last ViT blocks, V2 freezes the base GeoCLIP backbone and applies LoRA.

### Model experiment
- All base GeoCLIP weights are frozen.
- LoRA is applied to the ViT attention layers:
  - target modules: `q_proj`, `v_proj`
  - rank: **8**
  - alpha: **16**
  - dropout: **0.05**
- `visual_projection` and `image_encoder.mlp` are still unfrozen.
- Same custom regression head as V1.

### Loss and training
- Same basic loss as V1:
  - `0.05 * L1 loss + 0.95 * Haversine loss`
- Batch size: **16**
- Learning rate: **1e-4**
- Epochs: **30**
- Scheduler: cosine annealing
- AMP and gradient clipping used.

### Result shown in output
Best validation loss:
- **0.11775**

Evaluation output:
- Validation mean error: **120.5 km**
- Validation median error: **61.9 km**
- Test mean error: **122.2 km**
- Test median error: **63.5 km**
- Test within 100 km: **61.7%**
- Test within 500 km: **96.6%**

### Interpretation
V2 is a major improvement over V1. The LoRA strategy is more parameter-efficient and gives better validation/test distances than the partial-unfreezing approach.

---

## GeoClip_Training_V3.ipynb

### Purpose
This notebook switches the experiment from the smaller GLDv2-derived Italy dataset to the larger **OSV5M Italy dataset**.

### Dataset
- Uses OSV5M Italy metadata:
  - `train_metadata.csv`
  - `test_metadata.csv`
- Uses per-bucket TAR archives.
- Uses `NUM_BUCKETS = 20`.
- Runtime directories:
  - `/content/osv5m_images/train`
  - `/content/osv5m_images/test`

Available output:
- Train metadata rows: **111,787**
- Test metadata rows: **1,397**
- Train images found: **95,542 / 111,787**
- Test images found: **1,397 / 1,397**
- Trainable train+val pool: **95,542**
- Split:
  - Train: **81,210**
  - Validation: **14,332**
  - Test: **1,397**

### Main change from V2
The model strategy remains LoRA-based, but the dataset changes completely:
- from GLDv2 Italy dataset
- to OSV5M Italy street-view-like dataset

### Model experiment
- Same LoRA configuration as V2:
  - rank: **8**
  - alpha: **16**
  - targets: `q_proj`, `v_proj`
- Base GeoCLIP frozen.
- `visual_projection`, `image_encoder.mlp`, and regression head trainable.

### Training changes
- Batch size increased from **16** to **32**.
- Validation/test batch size uses multiplier:
  - `VAL_BATCH_MULTIPLIER = 4`
  - so val/test batch size is **128**
- Epochs reduced from **30** to **15**.
- Early stopping added:
  - patience: **5**
- Checkpoint frequency reduced to every **5** epochs to reduce Drive writes.

### Result shown in output
Visible training output reaches epoch 6 and part of epoch 7.

Best visible validation progress:
- Epoch 1 val: **154.8 km**
- Epoch 2 val: **132.1 km**
- Epoch 3 val: **122.2 km**
- Epoch 4 val: **118.3 km**
- Epoch 5 val: **110.7 km**
- Epoch 6 val: **104.8 km**

### Interpretation
V3 tests whether a much larger OSV5M Italy dataset improves generalization. The visible training curve suggests steady improvement, reaching better validation distance than V2 during early epochs.

---

## GeoClip_Training_V3_1.ipynb

### Purpose
This is an extension of V3 that adds a **land-aware penalty** to the loss function.

### Dataset
Same as V3:
- OSV5M Italy dataset
- 95,542 trainable train+val images
- 1,397 fixed test images

### Main change from V3
Adds a loss term that penalizes predictions far from land.

### Land-aware penalty
New functions added:
- `_build_land_penalty_grid`
- `land_penalty_loss`

New constants:
- `LAND_BUFFER_KM = 25.0`
- `LAND_SCALE_KM = 150.0`
- `LAND_PENALTY_WEIGHT = 0.20`

The penalty:
- is zero on land or near the coast
- increases for predictions deeper into the sea
- is computed using a grid over the Italy bounding box and bilinear interpolation

### Loss change
Previous V3 loss:
- `0.05 * L1 + 0.95 * Haversine`

New V3.1 loss:
- `0.05 * L1 + 0.95 * Haversine + 0.20 * land_penalty`

### Model
Same LoRA setup as V3:
- rank: **8**
- alpha: **16**
- targets: `q_proj`, `v_proj`

### Result shown in output
The visible outputs look very similar to V3 and show training through the same early epochs. The code change is clear, but the notebook output does not show a completed independent V3.1 result.

### Interpretation
V3.1 is an experiment to fix a geolocation-specific failure mode: the model may predict points in the sea or unrealistic off-land locations. The land penalty is designed to guide predictions back toward plausible land areas without completely forbidding coastal predictions.

---

## GeoClip_Training_V3_2.ipynb

### Purpose
This notebook increases the capacity of the LoRA adaptation and adds training optimization improvements.

### Dataset
Same OSV5M Italy dataset as V3/V3.1:
- Train metadata: **111,787**
- Test metadata: **1,397**
- Train found: **95,542**
- Test found: **1,397**
- Split:
  - Train: **81,210**
  - Validation: **14,332**
  - Test: **1,397**

### Main changes from V3.1

#### 1. LoRA capacity increased
V3.1:
- rank: **8**
- targets: `q_proj`, `v_proj`

V3.2:
- rank: **16**
- targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- alpha is set automatically to `2 * rank`, so alpha = **32**

This increases trainable LoRA parameters.

Visible output:
- Trainable parameters: **6,102,018 / 442,381,572**
- Trainable percentage: **1.38%**
- LoRA adapters: **3,145,728**

#### 2. Gradient accumulation added
- `GRAD_ACCUM_STEPS = 4`
- Batch size remains **32**
- Effective batch size becomes **128**

#### 3. Learning-rate warmup added
- `WARMUP_EPOCHS = 2`
- Scheduler changed to:
  - Linear warmup first
  - Then cosine annealing

#### 4. Checkpoint frequency changed
- `CHECKPOINT_FREQUENCY = 1`
- Saves checkpoint every epoch again.

#### 5. Optional torch.compile
- `USE_TORCH_COMPILE = True`
- Intended to improve throughput if available.

#### 6. Stratified validation split
The train/validation split is stratified by region, so the regional distribution is preserved.

Visible output:
- Train region distribution:
  - Nord: **46,768**
  - Sud/Isole: **18,755**
  - Centro: **15,687**
- Validation region distribution:
  - Nord: **8,254**
  - Sud/Isole: **3,310**
  - Centro: **2,768**

### Important output note
The visible output shows:
- `Device: cpu`
- warnings that CUDA is not available
- training starts extremely slowly

So the notebook appears to have been run without GPU at least in the visible execution.

### Interpretation
V3.2 is the strongest/most advanced training setup. It tests whether more LoRA capacity, stable larger effective batch size, warmup, and stratified splitting improve training stability and final accuracy. However, the visible run may not be useful unless it is rerun on GPU.

---

# Evaluation notebooks

## GeoClip_Evaluation_V1.ipynb

### Purpose
Evaluates baseline pretrained GeoCLIP against the fine-tuned model from the V1/V5 checkpoint family.

### Dataset
- Uses the GLDv2-derived Italy dataset.
- Uses same split logic as training.
- Available output:
  - Found: **34,183**
  - Split:
    - Train: **24,696**
    - Validation: **4,359**
    - Test: **5,128**

### Extra feature
This version includes a **sanity-check image section** with famous landmark images and hardcoded GPS metadata.

### Checkpoint
- Loads checkpoint from:
  - `checkpoints_v5/geoclip_italy_BEST.pth`
- Visible output says:
  - loaded from epoch 15
  - best validation loss around **0.16893**

### Evaluation result shown
Model comparison summary:

Baseline GeoCLIP:
- Mean error: **617.1 km**
- Median error: **216.0 km**
- Within 100 km: **33.3%**
- Within 500 km: **80.8%**

Fine-tuned GeoClipItaly:
- Mean error: **177.3 km**
- Median error: **168.0 km**
- Within 100 km: **32.3%**
- Within 500 km: **99.0%**

Classification accuracy:
- Baseline: **0.525**
- Fine-tuned: **0.626**

### Interpretation
The fine-tuned model strongly reduces mean error and improves broad-region classification accuracy, but it does not improve close-range accuracy in this particular evaluation.

---

## GeoClip_Evaluation_V2.ipynb

### Purpose
Evaluates the V2 LoRA fine-tuned model against baseline GeoCLIP.

### Dataset
- Uses the GLDv2-derived Italy dataset.
- Available output:
  - Found: **32,687**
  - Missing: **32,591**
  - Split:
    - Train: **23,615**
    - Validation: **4,168**
    - Test: **4,904**

### Main changes from Evaluation V1
- The sanity-check section was removed or simplified.
- Evaluation is more focused on the full test set.
- Uses checkpoint directory:
  - `checkpoints_v6`

### Checkpoint
- Loads:
  - `checkpoints_v6/geoclip_italy_BEST.pth`
- Visible output says:
  - loaded from epoch 27
  - best validation loss around **0.11837**

### Evaluation result shown
Baseline GeoCLIP:
- Mean error: **1027.8 km**
- Median error: **235.0 km**
- Within 100 km: **35.6%**
- Within 500 km: **66.0%**

Fine-tuned GeoClipItaly:
- Mean error: **171.1 km**
- Median error: **124.1 km**
- Within 100 km: **43.0%**
- Within 500 km: **96.4%**

Classification accuracy:
- Baseline: **0.679**
- Fine-tuned: **0.687**

### Interpretation
Evaluation V2 shows the LoRA fine-tuned model improves distance-based performance substantially, especially mean error and large-threshold accuracy. The classification accuracy improvement is small, but geolocation error improves clearly.

---

## GeoClip_Evaluation_V3.ipynb

### Purpose
Evaluates the OSV5M-trained model against baseline GeoCLIP.

### Dataset
- Uses OSV5M Italy dataset.
- Uses fixed test metadata.
- Available output:
  - Train rows: **111,787**
  - Test rows: **1,397**
  - Train found: **95,542 / 111,787**
  - Test found: **1,397 / 1,397**

### Main changes from Evaluation V2
- Dataset loading changed from GLDv2-style region TARs to OSV5M metadata and bucket TARs.
- Adds functions for:
  - computing macro-regions from coordinates
  - deploying train/test bucket archives
  - building image index
- Includes LoRA support through `peft`, matching the OSV5M training notebooks.

### Checkpoint
- Loads:
  - `checkpoints_v7/geoclip_italy_BEST.pth`
- Visible output says:
  - loaded from epoch 3
  - best validation loss around **0.11579**

### Evaluation result shown
Baseline GeoCLIP:
- Mean error: **1290.5 km**
- Median error: **538.3 km**
- Within 100 km: **14.0%**
- Within 500 km: **47.2%**

Fine-tuned GeoClipItaly:
- Mean error: **195.2 km**
- Median error: **124.2 km**
- Within 100 km: **42.2%**
- Within 500 km: **91.6%**

Classification accuracy:
- Baseline: **0.635**
- Fine-tuned: **0.764**

### Interpretation
Evaluation V3 shows that OSV5M fine-tuning gives a strong improvement over baseline on the OSV5M test set, especially in regional classification accuracy and median distance. The mean error remains higher than V2, likely because the OSV5M test set is more challenging/different.

---

# Summary table

| Notebook | Main experiment | Dataset | Key change | Main visible result |
|---|---|---|---|---|
| Training V1 | Partial backbone fine-tuning | GLDv2 Italy | Unfreeze last 2 ViT layers | Best val loss 0.16167, ~165 km |
| Training V2 | LoRA fine-tuning | GLDv2 Italy | LoRA rank 8 on q/v | Test mean 122.2 km, median 63.5 km |
| Training V3 | Larger dataset | OSV5M Italy | Switch dataset, batch 32, early stopping | Val improves to 104.8 km by epoch 6 |
| Training V3.1 | Plausibility constraint | OSV5M Italy | Add land-aware penalty | Code change clear; independent completed result not visible |
| Training V3.2 | Stronger LoRA + optimization | OSV5M Italy | LoRA rank 16, q/k/v/out, grad accumulation, warmup | Started on CPU in visible output; should rerun on GPU |
| Evaluation V1 | Baseline vs fine-tuned | GLDv2 Italy | Includes sanity images | FT mean 177.3 km vs baseline 617.1 km |
| Evaluation V2 | Baseline vs LoRA FT | GLDv2 Italy | Full test-set evaluation | FT mean 171.1 km vs baseline 1027.8 km |
| Evaluation V3 | Baseline vs OSV5M FT | OSV5M Italy | OSV5M loader/evaluation | FT mean 195.2 km vs baseline 1290.5 km |

---

# Final interpretation

The experimental history can be described like this:

> We started with direct partial fine-tuning of GeoCLIP on an Italy-only dataset. Then we replaced partial backbone unfreezing with LoRA, which gave a clear improvement while training fewer parameters. After that, we moved from the smaller GLDv2-derived Italy dataset to the larger OSV5M Italy dataset to improve coverage and realism. The next experiment added a land-aware penalty to discourage unrealistic sea predictions. Finally, the latest version increased LoRA capacity and introduced training-stability improvements such as gradient accumulation, warmup scheduling, region-stratified splitting, and optional compilation.

The best complete result visible in the uploaded notebooks is from **Training V2 / Evaluation V2** on the GLDv2-derived Italy dataset, with a test mean error around **122 km** in the training notebook and **171 km** in the separate evaluation notebook. The OSV5M experiments are promising, but V3.1 and V3.2 need a clean completed GPU run to compare fairly.
