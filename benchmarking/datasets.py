"""Dataset wrappers and OSV5M deploy helpers."""

from __future__ import annotations

import os
import shutil
import tarfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from .config import CLIP_MEAN, CLIP_STD, MAX_LAT, MAX_LON, MIN_LAT, MIN_LON, NUM_BUCKETS, BenchmarkConfig
from .spatial import ItalySpatialFilter


@dataclass
class GeolocSample:
    image_id: str
    image_path: str
    lat: float
    lon: float
    source: str
    extra: Dict[str, Any]


def image_subdir(image_id: str) -> str:
    return f"{int(str(image_id)) % NUM_BUCKETS:02d}"


def resolve_image_path(image_id: str, runtime_dir: str) -> str:
    sid = image_subdir(image_id)
    return os.path.join(runtime_dir, sid, f"{image_id}.jpg")


def load_metadata_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = ItalySpatialFilter.normalize_columns(df)
    df["image_id"] = df["image_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["image_id", "lat", "lon"]).reset_index(drop=True)
    return df


def build_image_index(meta_df: pd.DataFrame, runtime_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for img_id in meta_df["image_id"]:
        path = ""
        if "local_path" in meta_df.columns:
            rows = meta_df.loc[meta_df["image_id"] == img_id, "local_path"]
            if len(rows):
                path = rows.iloc[0]
        if isinstance(path, str) and path and os.path.exists(path):
            index[img_id] = path
            continue
        candidate = resolve_image_path(img_id, runtime_dir)
        if os.path.exists(candidate):
            index[img_id] = candidate
    return index


def _make_subdir_tar(src_dir: str, tar_path: str) -> int:
    images = list(Path(src_dir).glob("*.jpg"))
    if not images:
        return 0
    if os.path.exists(tar_path):
        tar_mtime = os.path.getmtime(tar_path)
        newest_img = max(p.stat().st_mtime for p in images)
        if tar_mtime >= newest_img:
            return len(images)
    with tarfile.open(tar_path, "w") as tar:
        for img in images:
            tar.add(str(img), arcname=img.name)
    return len(images)


def deploy_split(
    split_name: str,
    drive_image_dir: str,
    tar_cache_dir: str,
    runtime_dir: str,
) -> None:
    """Copy bucket TARs from Drive to runtime (idempotent)."""
    os.makedirs(tar_cache_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)
    buckets = sorted(
        d for d in os.listdir(drive_image_dir)
        if os.path.isdir(os.path.join(drive_image_dir, d)) and d.isdigit()
    )
    if not buckets:
        raise RuntimeError(
            f"[{split_name}] No bucket subdirs in {drive_image_dir}. "
            "Run osv5m/osv5m_dataset_preprocess.ipynb first."
        )
    t_total = time.perf_counter()
    total_images = 0
    for bucket in tqdm(buckets, desc=f"deploy {split_name}", unit="bucket"):
        src_dir = os.path.join(drive_image_dir, bucket)
        tar_path = os.path.join(tar_cache_dir, f"{bucket}.tar")
        tmp_tar = f"/content/_osv5m_{split_name}_{bucket}.tar"
        dest_dir = os.path.join(runtime_dir, bucket)
        os.makedirs(dest_dir, exist_ok=True)
        src_images = list(Path(src_dir).glob("*.jpg"))
        existing = list(Path(dest_dir).glob("*.jpg"))
        if src_images and len(existing) >= len(src_images):
            total_images += len(existing)
            continue
        n = _make_subdir_tar(src_dir, tar_path)
        if n == 0:
            continue
        shutil.copy2(tar_path, tmp_tar)
        with tarfile.open(tmp_tar, "r") as tar:
            tar.extractall(dest_dir)
        total_images += n
        os.remove(tmp_tar)
    elapsed = time.perf_counter() - t_total
    print(f"[{split_name}] {total_images:,} images ready in {elapsed:.1f}s")


class BaseGeolocDataset(Dataset, ABC):
    """Abstract geolocalization dataset returning (tensor, target, meta)."""

    source_name: str = "base"

    def __init__(
        self,
        transform: Optional[Callable] = None,
        require_image: bool = True,
    ) -> None:
        self.transform = transform
        self.require_image = require_image
        self.samples: List[GeolocSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def _load_image(self, sample: GeolocSample) -> Image.Image:
        ...

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        lat = max(MIN_LAT, min(MAX_LAT, sample.lat))
        lon = max(MIN_LON, min(MAX_LON, sample.lon))
        norm_lat = (lat - MIN_LAT) / (MAX_LAT - MIN_LAT)
        norm_lon = (lon - MIN_LON) / (MAX_LON - MIN_LON)
        target = torch.tensor([norm_lat, norm_lon], dtype=torch.float32)
        meta = {
            "image_id": sample.image_id,
            "image_path": sample.image_path,
            "lat": lat,
            "lon": lon,
            "source": sample.source,
            **sample.extra,
        }
        if self.transform is None:
            return None, target, meta
        img = self._load_image(sample)
        return self.transform(img), target, meta


class OSV5MBenchmarkDataset(BaseGeolocDataset):
    """OSV5M Italy test split (fixed test_metadata.csv, 1 km train separation by design)."""

    source_name = "osv5m"

    def __init__(
        self,
        meta_df: pd.DataFrame,
        image_index: Dict[str, str],
        transform: Optional[Callable] = None,
        n_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(transform=transform, require_image=True)
        df = meta_df.copy()
        df = df[df["image_id"].isin(image_index)].reset_index(drop=True)
        if n_samples is not None and len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
        for _, row in df.iterrows():
            img_id = str(row["image_id"])
            self.samples.append(
                GeolocSample(
                    image_id=img_id,
                    image_path=image_index[img_id],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    source=self.source_name,
                    extra={"region": row.get("region", "")},
                )
            )

    def _load_image(self, sample: GeolocSample) -> Image.Image:
        if not sample.image_path or not os.path.exists(sample.image_path):
            raise FileNotFoundError(f"Missing image: {sample.image_id} -> {sample.image_path}")
        return Image.open(sample.image_path).convert("RGB")


class CSVOpenDomainDataset(BaseGeolocDataset):
    """Generic CSV benchmark (IM2GPS / YFCC) with Italy spatial filter."""

    def __init__(
        self,
        name: str,
        csv_path: str,
        image_root: str,
        spatial_filter: ItalySpatialFilter,
        transform: Optional[Callable] = None,
        path_columns: Tuple[str, ...] = ("image_path", "filepath", "filename", "img_path"),
        n_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(transform=transform, require_image=True)
        self.source_name = name
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{name} metadata not found at {csv_path}. "
                f"Place Italy-filtered images under {image_root}."
            )
        raw = pd.read_csv(csv_path)
        raw = ItalySpatialFilter.normalize_columns(raw)
        filtered = spatial_filter.filter_dataframe(raw, desc=f"{name} Italy filter")
        if n_samples is not None and len(filtered) > n_samples:
            filtered = filtered.sample(n=n_samples, random_state=random_seed)

        for _, row in filtered.iterrows():
            img_id = str(row.get("image_id", row.name))
            rel = None
            for col in path_columns:
                if col in row and pd.notna(row[col]):
                    rel = str(row[col])
                    break
            if rel is None:
                continue
            full_path = rel if os.path.isabs(rel) else os.path.join(image_root, rel)
            if not os.path.exists(full_path):
                continue
            self.samples.append(
                GeolocSample(
                    image_id=img_id,
                    image_path=full_path,
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    source=name,
                    extra={},
                )
            )
        if not self.samples:
            raise RuntimeError(
                f"No usable {name} samples after Italy filter and path resolution. "
                f"Check {csv_path} and image_root={image_root}"
            )

    def _load_image(self, sample: GeolocSample) -> Image.Image:
        return Image.open(sample.image_path).convert("RGB")


def default_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])


def build_tta_transforms() -> List[transforms.Compose]:
    mean, std = list(CLIP_MEAN), list(CLIP_STD)
    norm = transforms.Normalize(mean=mean, std=std)
    return [
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm]),
        transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]),
        transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm,
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), norm,
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomRotation(10),
            transforms.ToTensor(), norm,
        ]),
    ]


def prepare_osv5m_test_dataset(cfg: BenchmarkConfig, spatial_filter: ItalySpatialFilter) -> OSV5MBenchmarkDataset:
    deploy_split("test", cfg.osv5m_test_image_dir, cfg.osv5m_test_tar_dir, cfg.runtime_test_dir)
    meta = load_metadata_csv(cfg.osv5m_test_meta_csv)
    meta = spatial_filter.filter_dataframe(meta, lat_col="lat", lon_col="lon", desc="OSV5M test Italy")
    index = build_image_index(meta, cfg.runtime_test_dir)
    print(f"OSV5M test: {len(index):,} / {len(meta):,} images found on disk")
    return OSV5MBenchmarkDataset(
        meta_df=meta,
        image_index=index,
        transform=default_eval_transform(),
        n_samples=cfg.n_test_samples,
        random_seed=cfg.random_seed,
    )


def prepare_im2gps_dataset(cfg: BenchmarkConfig, spatial_filter: ItalySpatialFilter) -> CSVOpenDomainDataset:
    imgdir = getattr(cfg, "im2gps_imgdir", "im2gps3ktest")
    csv_path = os.path.join(cfg.im2gps_root, "im2gps_italy.csv")
    image_root = os.path.join(cfg.im2gps_root, imgdir)
    return CSVOpenDomainDataset(
        name="im2gps",
        csv_path=csv_path,
        image_root=image_root,
        spatial_filter=spatial_filter,
        transform=default_eval_transform(),
        n_samples=cfg.n_test_samples,
        random_seed=cfg.random_seed,
    )


def prepare_yfcc_dataset(cfg: BenchmarkConfig, spatial_filter: ItalySpatialFilter) -> CSVOpenDomainDataset:
    imgdir = getattr(cfg, "yfcc_imgdir", "images")
    csv_path = os.path.join(cfg.yfcc_root, "yfcc26k_italy.csv")
    image_root = os.path.join(cfg.yfcc_root, imgdir)
    return CSVOpenDomainDataset(
        name="yfcc26k",
        csv_path=csv_path,
        image_root=image_root,
        spatial_filter=spatial_filter,
        transform=default_eval_transform(),
        n_samples=cfg.n_test_samples,
        random_seed=cfg.random_seed,
    )


def build_dataloader(
    dataset: BaseGeolocDataset,
    batch_size: int,
    num_workers: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    def collate(batch):
        imgs, targets, metas = zip(*batch)
        if imgs[0] is None:
            return None, torch.stack(targets), list(metas)
        return torch.stack(imgs), torch.stack(targets), list(metas)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
