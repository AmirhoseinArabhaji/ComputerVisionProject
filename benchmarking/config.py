"""Shared configuration for the Italy geolocalization benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Italy bounding box — must match training / osv5m preprocessing.
MIN_LAT, MAX_LAT = 35.4, 47.2
MIN_LON, MAX_LON = 6.6, 18.8

# GeoCLIP paper (NeurIPS 2023) standard thresholds — see geoclip/train/eval.py
PAPER_THRESHOLDS_KM = (1, 25, 200, 2500)

# Extended thresholds for Italy fine-tuning analysis.
ACCURACY_THRESHOLDS_KM = (1, 10, 25, 50, 100, 200, 500, 2500)

# CLIP normalization (OpenAI ViT-L/14).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# OSV5M bucket layout.
NUM_BUCKETS = 20

# OSV5M test split is built with >=1 km separation from train (dataset design).
OSV5M_TEST_MIN_TRAIN_SEPARATION_KM = 1.0

# Bump when breaking changes require a full benchmarking/ folder re-sync on Drive.
BENCHMARK_PACKAGE_VERSION = 7


@dataclass
class BenchmarkConfig:
    """Runtime paths and inference options (Colab / local)."""

    drive_base: str = "/content/drive/MyDrive/Vision_Project_2026"
    checkpoint_dir: str = ""
    checkpoint_filename: str = "geoclip_italy_BEST.pth"
    osv5m_base: str = ""
    runtime_test_dir: str = "/content/osv5m_images/test"
    runtime_train_dir: str = "/content/osv5m_images/train"

    # Open-domain benchmarks (optional).
    im2gps_root: str = ""
    yfcc_root: str = ""

    batch_size: int = 32
    num_workers: int = 2
    n_test_samples: Optional[int] = None
    random_seed: int = 42

    use_tta: bool = True
    n_tta_views: int = 6
    lora_rank: int = 16
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")

    use_polygon_filter: bool = True
    poly_buffer_deg: float = 0.02
    constrain_baselines_to_italy: bool = True
    # False: global GeoCLIP gallery (not Italy-constrained) — set True only for ablation.
    include_global_geoclip_baseline: bool = False
    # Extracted image subfolder name inside im2gps_root (zip extracts to im2gps3ktest/).
    im2gps_imgdir: str = "im2gps3ktest"
    # Extracted image subfolder name inside yfcc_root.
    yfcc_imgdir: str = "images"

    # True: add TIBHannover/GeoEstimation (ISNs equivalent from GeoCLIP paper comparisons).
    # Requires: pip install git+https://github.com/TIBHannover/GeoEstimation.git
    include_geoestimation: bool = False
    # True: add StreetCLIP (Haas et al. 2023, geolocal/StreetCLIP on HuggingFace).
    # CLIP ViT-L/14 fine-tuned on geolocalized images; Italy GPS retrieval via text gallery.
    # No extra install needed (transformers is a dependency of geoclip).
    include_streetclip: bool = False

    output_dir: str = "/content/benchmark_results"
    datasets: tuple[str, ...] = ("osv5m",)

    def __post_init__(self) -> None:
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"{self.drive_base}/checkpoints_v3_3"
        if not self.osv5m_base:
            self.osv5m_base = f"{self.drive_base}/osv5m/osv5m_italy_dataset"
        if not self.im2gps_root:
            self.im2gps_root = f"{self.drive_base}/benchmarks/im2gps"
        if not self.yfcc_root:
            self.yfcc_root = f"{self.drive_base}/benchmarks/yfcc26k"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.checkpoint_filename)

    @property
    def osv5m_test_meta_csv(self) -> str:
        return os.path.join(self.osv5m_base, "test_images", "test_metadata.csv")

    @property
    def osv5m_train_meta_csv(self) -> str:
        return os.path.join(self.osv5m_base, "train_images", "train_metadata.csv")

    @property
    def osv5m_test_image_dir(self) -> str:
        return os.path.join(self.osv5m_base, "test_images")

    @property
    def osv5m_train_image_dir(self) -> str:
        return os.path.join(self.osv5m_base, "train_images")

    @property
    def osv5m_test_tar_dir(self) -> str:
        return os.path.join(self.osv5m_base, "test_subdirtars")

    @property
    def osv5m_train_tar_dir(self) -> str:
        return os.path.join(self.osv5m_base, "train_subdirtars")

    def ensure_output_dir(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir


def check_package_version(min_version: int = 4) -> None:
    """Raise if Drive has a stale partial copy of the benchmarking package."""
    ver = globals().get("BENCHMARK_PACKAGE_VERSION", 0)
    if ver < min_version or "PAPER_THRESHOLDS_KM" not in globals():
        raise RuntimeError(
            f"Stale benchmarking/ on Drive (version={ver}, need>={min_version}). "
            "Re-copy the entire benchmarking/ folder from the repo, then "
            "Runtime -> Restart session and re-run from the config cell."
        )
    print(f"benchmarking package version: {ver}")
