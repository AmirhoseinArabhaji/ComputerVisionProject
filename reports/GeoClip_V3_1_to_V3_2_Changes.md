# GeoClip Training V3.2 — Changes vs V3.1

## Summary

V3.2 adds five training improvements on top of V3.1. The loss function, dataset, and
regression head are unchanged. Checkpoint format is backward-compatible (strict=False
loading still works) but the LoRA architecture change means LoRA weights from V3.1
cannot be reused — only the regressor/visual_projection/mlp weights carry over.

---

## 1. Expanded LoRA Target Modules + Rank 16

**V3.1**
```python
LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'])
```

**V3.2**
```python
LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK*2, target_modules=LORA_TARGET_MODULES)
# defaults: LORA_RANK=16, LORA_TARGET_MODULES=['q_proj', 'k_proj', 'v_proj', 'out_proj']
```

**Why**: Targeting only Q and V leaves the key-routing (`k_proj`) and output-mixing
(`out_proj`) layers unchanged. These are linear layers in the same attention block and
contribute equally to domain shift. Adding them doubles the adapter's representational
coverage. Increasing rank from 8 to 16 doubles adapter capacity without meaningful VRAM
cost (~3.1M LoRA params vs ~0.8M). `alpha = 2 * rank` is the standard initialization
that keeps the effective learning rate of the adapter invariant to rank.

**Checkpoint compatibility**: LoRA weight shapes change (rank 8→16, 2→4 targets), so
LoRA keys from a V3.1 checkpoint will be rejected on strict=False load — LoRA restarts
from random init. Non-LoRA weights (regressor, visual_projection, mlp) load normally
and provide a warm start. Use `checkpoints_v8` to avoid accidentally loading V3.1 LoRA
weights.

---

## 2. LR Warmup (LinearLR → CosineAnnealingLR)

**V3.1**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
```

**V3.2**
```python
_warmup = LinearLR(optimizer, start_factor=1/WARMUP_EPOCHS, end_factor=1.0,
                   total_iters=WARMUP_EPOCHS)                      # default: 2 epochs
_cosine = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS-WARMUP_EPOCHS, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[_warmup, _cosine],
                         milestones=[WARMUP_EPOCHS])
```

**Why**: Starting at full LR immediately can distort the frozen CLIP embedding space
before the LoRA adapters have stabilized. A 2-epoch linear ramp from `LR/2` to `LR`
allows the regressor and domain-adaptation layers to settle first. After warmup, cosine
annealing proceeds as before over the remaining epochs.

**New config key**: `WARMUP_EPOCHS = 2`

---

## 3. Gradient Accumulation

**V3.1** — optimizer step every batch (effective batch = 32).

**V3.2**
```python
GRAD_ACCUM_STEPS = 4   # effective batch = 32 * 4 = 128

loss_scaled = loss / GRAD_ACCUM_STEPS
scaler.scale(loss_scaled).backward()

if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

**Why**: Larger effective batch size stabilizes gradient estimates, especially useful
with the noisier haversine + land-penalty loss. T4 VRAM cannot fit batch=128 directly;
accumulation achieves the same gradient statistics with no memory overhead.

**Note on LR**: LR is NOT auto-scaled with GRAD_ACCUM_STEPS. If you increase steps
significantly, consider the linear scaling rule: `new_LR = LR * GRAD_ACCUM_STEPS`.

**Logged loss**: `train_loss` accumulates the unscaled `loss.item()` (before dividing
by GRAD_ACCUM_STEPS) so epoch-level loss values remain comparable with V3.1.

---

## 4. Stratified Train/Val Split by Region

**V3.1**
```python
train_df, val_df = train_test_split(trainable_df, test_size=VAL_SIZE,
                                    random_state=RANDOM_SEED)
```

**V3.2**
```python
train_df, val_df = train_test_split(trainable_df, test_size=VAL_SIZE,
                                    random_state=RANDOM_SEED,
                                    stratify=trainable_df['region'])
```

**Why**: The OSV5M Italy train pool is heavily Nord-biased (Nord 58% / Sud_Isole 23% /
Centro 19%). Without stratification a random split can over-represent Nord in val and
under-represent Sud/Isole, making val loss a poor proxy for generalization across Italy.
`stratify` forces each split to mirror the full distribution.

The region labels (`nord` / `centro` / `sud_isole`) are derived from latitude at
metadata load time and exist as a column in `trainable_df` — no structural change to
the dataset pipeline is needed.

---

## 5. DataLoader `prefetch_factor=4`

**V3.1** — prefetch_factor defaulted to 2.

**V3.2**
```python
DataLoader(..., prefetch_factor=4)   # all three loaders
```

**Why**: Images are loaded from `/content/osv5m_images/` (Drive→runtime copy), which
is slower than a local NVMe. With num_workers=2 and prefetch_factor=4, each worker
pre-loads 4 batches, giving the GPU a 8-batch buffer. This reduces the chance of the
GPU stalling on data I/O between batches.

---

## 6. torch.compile (optional, best-effort)

**V3.1** — not present.

**V3.2**
```python
USE_TORCH_COMPILE = True   # set False to skip

if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model, mode='default')
    except Exception as e:
        print(f"torch.compile skipped ({e}).")
```

**Why**: `torch.compile` fuses kernels and eliminates Python dispatch overhead,
typically giving 10–30% throughput improvement on T4. Applied **after** checkpoint
loading (state_dict keys are unaffected by compile). The first training batch
triggers compilation (~30 s); subsequent batches run at compiled speed. The try/except
ensures the notebook runs normally on Colab environments where compile is unavailable.

**Mode `default` not `reduce-overhead`**: `reduce-overhead` captures CUDAGraphs which
require static memory addresses across replays. PEFT/LoRA layers compute
`result += lora_B(lora_A(dropout(x))) * scaling` — an in-place addition that writes
into a tensor whose address the CUDAGraph already captured for a prior replay. This
causes `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run`. `default` mode still fuses via TorchInductor but does not use
CUDAGraphs, avoiding the conflict entirely.

---

## Checkpoint Directory

| Version | Dir |
|---|---|
| V3.1 | `checkpoints_v3_1` |
| V3.2 | `checkpoints_v3_2` |

Separate directory prevents accidentally loading V3.1 LoRA weights (which have
incompatible shapes due to rank and target changes) as the "latest checkpoint".

---

## What Was NOT Changed

| Item | Reason |
|---|---|
| Regression head (`512→1024→512→256→2`) | Changing would break weight loading from any prior checkpoint; user uncertain about benefit |
| Loss function (L1 + Haversine + land penalty) | No improvement identified |
| Image augmentation pipeline | Matches inference preprocessing — changing risks train/test mismatch |
| Bounding box `[35.4,47.2] × [6.6,18.8]` | Fixed by dataset |
| Batch size (32) | T4 VRAM limit; effective batch addressed via grad accumulation |
| Dataset pipeline (OSV5M, 20 buckets, deploy_split) | Unchanged; region stratification handled at metadata level only |
