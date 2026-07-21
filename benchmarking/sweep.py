"""Cross-version consistency sweep.

Evaluates multiple fine-tuned checkpoints against one shared set of baselines, on the
same test set, through the same inference/metrics code path. This exists because the
per-version training-notebook numbers are NOT directly comparable: V1/V2 report val
loss on GLDv2, V3.1+ report val loss on OSV5M, and the two are different distributions
with different difficulty profiles. Running every OSV5M-trained checkpoint through this
sweep instead gives one consistent held-out test-set metric (median km, accuracy@threshold)
per version.

V1 and V2 are intentionally excluded from `default_version_specs`: V1 uses a different
architecture (no LoRA, a smaller regression head) that `GeoClipItaly` cannot load, and V2
-- while architecturally loadable -- was trained on GLDv2, not OSV5M, so its numbers here
would measure cross-dataset transfer rather than an apples-to-apples ablation step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch

from .config import BenchmarkConfig
from .datasets import build_dataloader
from .metrics import compute_errors_km, compute_metrics
from .predictors import GeoCLIPRetrievalPredictor, GeoClipItalyPredictor, StreetCLIPPredictor
from .runner import build_dataset
from .spatial import ItalySpatialFilter


@dataclass
class VersionSpec:
    """One fine-tuned checkpoint to include in the cross-version sweep."""

    name: str
    checkpoint_dir: str
    lora_rank: int
    lora_target_modules: Sequence[str]
    checkpoint_filename: str = "geoclip_italy_BEST.pth"
    use_tta: bool = False  # off by default so the sweep isolates training changes from TTA


def default_version_specs(drive_base: str) -> List[VersionSpec]:
    return [
        VersionSpec("V3", f"{drive_base}/checkpoints_v3", lora_rank=8, lora_target_modules=("q_proj", "v_proj")),
        VersionSpec("V3.1", f"{drive_base}/checkpoints_v3_1", lora_rank=8, lora_target_modules=("q_proj", "v_proj")),
        VersionSpec(
            "V3.2", f"{drive_base}/checkpoints_v3_2", lora_rank=16,
            lora_target_modules=("q_proj", "k_proj", "v_proj", "out_proj"),
        ),
        VersionSpec(
            "V3.3", f"{drive_base}/checkpoints_v3_3", lora_rank=16,
            lora_target_modules=("q_proj", "k_proj", "v_proj", "out_proj"),
        ),
        VersionSpec(
            "V3.3 + TTA", f"{drive_base}/checkpoints_v3_3", lora_rank=16,
            lora_target_modules=("q_proj", "k_proj", "v_proj", "out_proj"), use_tta=True,
        ),
    ]


def run_checkpoint_sweep(
    cfg: BenchmarkConfig,
    version_specs: Optional[List[VersionSpec]] = None,
    dataset_name: str = "osv5m",
    include_italy_baseline: bool = True,
    include_global_baseline: bool = True,
    include_streetclip: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Returns {model_name: predictions_df}, ready for the existing plot_*/table helpers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_filter = ItalySpatialFilter(use_polygon=cfg.use_polygon_filter, poly_buffer_deg=cfg.poly_buffer_deg)
    dataset = build_dataset(cfg, dataset_name, spatial_filter)
    loader = build_dataloader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    print(f"[sweep] {dataset_name}: {len(dataset):,} samples")

    results: Dict[str, pd.DataFrame] = {}

    def _run(predictor) -> None:
        rows = predictor.run_on_dataloader(loader)
        df = pd.DataFrame(rows).dropna(subset=["lat_true", "lon_true"])
        valid = compute_errors_km(df) if not df.empty else df
        results[predictor.name] = valid if not valid.empty else df
        compute_metrics(results[predictor.name]).pretty_print(predictor.name)

    if include_italy_baseline:
        _run(GeoCLIPRetrievalPredictor(device=device, italy_constrained=True))
    if include_global_baseline:
        _run(GeoCLIPRetrievalPredictor(device=device, italy_constrained=False))
    if include_streetclip:
        _run(StreetCLIPPredictor(device=device, spatial_filter=spatial_filter, train_meta_csv=cfg.osv5m_train_meta_csv))

    for spec in version_specs if version_specs is not None else default_version_specs(cfg.drive_base):
        ckpt_path = os.path.join(spec.checkpoint_dir, spec.checkpoint_filename)
        if not os.path.exists(ckpt_path):
            print(f"[sweep] SKIPPED {spec.name}: checkpoint not found at {ckpt_path}")
            continue
        predictor = GeoClipItalyPredictor(
            checkpoint_path=ckpt_path,
            device=device,
            lora_rank=spec.lora_rank,
            lora_target_modules=spec.lora_target_modules,
            use_tta=spec.use_tta,
            n_tta_views=cfg.n_tta_views,
        )
        predictor.name = spec.name
        _run(predictor)

    return results
