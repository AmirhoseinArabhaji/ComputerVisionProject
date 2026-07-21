"""Prepare open-domain Italy benchmark subsets (IM2GPS3k, YFCC26k).

Download instructions
---------------------
IM2GPS3k (Zhu et al., CVPR 2016):
  http://www.mediafire.com/file/7ht7nc4rk29bpsa/im2gps3ktest.zip
  or: https://github.com/lugiavn/revisiting-im2gps/releases
  Place as: {im2gps_root}/im2gps3k_test.csv   (columns: id/filename, lat, lon)
             {im2gps_root}/images/*.jpg

YFCC26k (GeoCLIP paper release):
  https://github.com/ViktorooReps/geoclip  (see Datasets in README)
  Place as: {yfcc_root}/yfcc26k_test.csv   (columns: photo_id, latitude, longitude)
             {yfcc_root}/images/*.jpg

Expected Italy yield (approx):
  IM2GPS3k  : ~20-60 images  (0.7-2% of 2,997 total)
  YFCC26k   : ~150-400 images (0.6-1.5% of 26,280 total)

These are secondary benchmarks; OSV5M Italy (1,397 images) is the primary one.
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import List, Optional

import pandas as pd

from .spatial import ItalySpatialFilter

# ---------------------------------------------------------------------------
# Candidate filenames searched in order
# ---------------------------------------------------------------------------

_IM2GPS_CSV_CANDIDATES: List[str] = [
    "im2gps3k_test.csv",
    "im2gps3ktest.csv",
    "im2gps_test.csv",
    "im2gps3k_places365.csv",
    "im2gps3k.csv",
    "testset_gt.csv",
    "im2gps.csv",
    "test.csv",
]

_YFCC_CSV_CANDIDATES: List[str] = [
    "yfcc26k_test.csv",
    "yfcc26k.csv",
    "yfcc_test.csv",
    "yfcc.csv",
    "test.csv",
]

# Column aliases for auto-detection (case-sensitive, tried in set membership)
_LAT_ALIASES = {"lat", "latitude", "LAT", "Latitude", "LATITUDE", "lat_deg", "lat_y"}
_LON_ALIASES = {"lon", "longitude", "lng", "long", "LON", "LONG", "Longitude", "LONGITUDE", "lon_x"}
_ID_ALIASES  = {"id", "image_id", "photo_id", "img_id", "IMG_ID", "filename", "image", "name", "ID"}

_MIN_SAMPLES_WARN = 10  # warn when Italy subset is very small


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_extract_zip(root: str, imgdir: str) -> bool:
    """
    If `imgdir` inside `root` is missing or empty, find any .zip in `root` and extract it.
    Returns True if images are available after the attempt.
    """
    import zipfile as _zf

    imgdir_path = os.path.join(root, imgdir)
    if os.path.isdir(imgdir_path) and len(os.listdir(imgdir_path)) > 10:
        return True  # already extracted

    zips = [f for f in os.listdir(root) if f.lower().endswith(".zip")]
    if not zips:
        return False

    for fname in zips:
        zip_path = os.path.join(root, fname)
        print(f"[open_domain] Auto-extracting {zip_path} ...")
        try:
            with _zf.ZipFile(zip_path, "r") as zf:
                zf.extractall(root)
        except Exception as exc:
            print(f"  WARNING: extraction failed: {exc}")
            continue
        if os.path.isdir(imgdir_path) and len(os.listdir(imgdir_path)) > 0:
            print(f"  Extracted {len(os.listdir(imgdir_path)):,} files -> {imgdir_path}")
            return True

    return os.path.isdir(imgdir_path) and len(os.listdir(imgdir_path)) > 0


def _find_csv(root: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return None


def _sniff_sep(csv_path: str) -> str:
    """Detect delimiter (comma, tab, pipe, semicolon) from the first 8 KB."""
    try:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        return ","


def _load_and_normalize(csv_path: str) -> pd.DataFrame:
    """Load CSV/TSV (auto-detects delimiter) and rename lat/lon/image_id columns."""
    sep = _sniff_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    rename: dict[str, str] = {}
    for col in df.columns:
        s = col.strip()
        if s in _LAT_ALIASES and "lat" not in rename.values():
            rename[col] = "lat"
        elif s in _LON_ALIASES and "lon" not in rename.values():
            rename[col] = "lon"
        elif s in _ID_ALIASES and "image_id" not in rename.values():
            rename[col] = "image_id"
    df = df.rename(columns=rename)

    missing = [c for c in ("lat", "lon") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot map lat/lon columns in {csv_path}.\n"
            f"  Found   : {list(df.columns)}\n"
            f"  Missing : {missing}\n"
            "  Rename them to 'lat'/'lon' in the CSV and re-run."
        )
    if "image_id" not in df.columns:
        df["image_id"] = df.index.astype(str)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def _attach_image_paths(df: pd.DataFrame, image_root: str) -> pd.DataFrame:
    """
    Add image_path column; rows without an existing file on disk are kept
    (CSVOpenDomainDataset handles missing files gracefully at runtime).
    """
    if not os.path.isdir(image_root):
        warnings.warn(
            f"Image directory not found: {image_root}. "
            "CSVOpenDomainDataset will look up files by image_id at inference time."
        )
        df = df.copy()
        df["image_path"] = df["image_id"].apply(lambda x: os.path.join(image_root, f"{x}.jpg"))
        return df

    rows = []
    found = 0
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        # Handle image IDs that already include a file extension (e.g. IM2GPS3k IMG_ID column)
        _ext = os.path.splitext(img_id)[1].lower()
        if _ext in (".jpg", ".jpeg", ".png"):
            _base = os.path.splitext(img_id)[0]
            candidates = [
                os.path.join(image_root, img_id),           # exact match
                os.path.join(image_root, f"{_base}.jpg"),
                os.path.join(image_root, f"{_base}.jpeg"),
                os.path.join(image_root, f"{_base}.png"),
            ]
        else:
            candidates = [
                os.path.join(image_root, f"{img_id}.jpg"),
                os.path.join(image_root, f"{img_id}.jpeg"),
                os.path.join(image_root, f"{img_id}.png"),
            ]
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        r = row.to_dict()
        r["image_path"] = path
        if os.path.exists(path):
            found += 1
        rows.append(r)

    print(f"  Images on disk  : {found:,} / {len(df):,}")
    return pd.DataFrame(rows).reset_index(drop=True)


def _ensure_italy_csv(
    root: str,
    source_name: str,
    candidates: List[str],
    output_name: str,
    spatial_filter: ItalySpatialFilter,
    image_subdir: str = "images",
) -> str:
    out_path = os.path.join(root, output_name)

    if os.path.exists(out_path):
        cached = pd.read_csv(out_path)
        print(f"[{source_name}] Cached Italy CSV: {len(cached):,} rows -> {out_path}")
        return out_path

    src = _find_csv(root, candidates)
    if src is None:
        raise FileNotFoundError(
            f"[{source_name}] No source CSV found in {root}.\n"
            f"  Searched: {candidates}\n"
            "  Download the dataset (see benchmarking/open_domain.py docstring) and re-run."
        )

    print(f"[{source_name}] Loading {src} ...")
    raw = _load_and_normalize(src)
    print(f"[{source_name}]   Total rows  : {len(raw):,}")

    italy = spatial_filter.filter_dataframe(raw, desc=f"{source_name} Italy filter")
    n = len(italy)
    print(f"[{source_name}]   Italy rows  : {n:,}")

    if n < _MIN_SAMPLES_WARN:
        warnings.warn(
            f"[{source_name}] Only {n} Italy samples — too few for reliable statistics. "
            "Use OSV5M Italy (~1,397 images) as the primary benchmark."
        )

    image_root = os.path.join(root, image_subdir)
    italy = _attach_image_paths(italy, image_root)
    italy.to_csv(out_path, index=False)
    print(f"[{source_name}]   Saved       -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_im2gps_italy_csv(
    im2gps_root: str,
    spatial_filter: ItalySpatialFilter,
    image_subdir: str = "im2gps3ktest",
) -> str:
    """Filter IM2GPS3k to Italy polygon and cache as im2gps_italy.csv.

    Args:
        im2gps_root:   Dir containing im2gps3k_places365.csv and im2gps3ktest.zip.
        spatial_filter: Configured ItalySpatialFilter instance.
        image_subdir:  Sub-directory with JPEG files (default: "im2gps3ktest").

    Returns:
        Path to the cached Italy CSV.
    """
    out_path = os.path.join(im2gps_root, "im2gps_italy.csv")

    # Invalidate cached CSV if image paths inside it no longer exist on disk.
    # This happens when the CSV was built before the zip was extracted.
    if os.path.exists(out_path):
        try:
            cached = pd.read_csv(out_path)
            if "image_path" in cached.columns and len(cached) > 0:
                sample_paths = cached["image_path"].dropna().head(5).tolist()
                if sample_paths and not any(os.path.exists(str(p)) for p in sample_paths):
                    print("[IM2GPS3k] Cached CSV has stale image paths — will regenerate after extraction.")
                    os.remove(out_path)
        except Exception:
            pass

    # Auto-extract zip if the image directory is missing or empty.
    _auto_extract_zip(im2gps_root, image_subdir)

    return _ensure_italy_csv(
        root=im2gps_root,
        source_name="IM2GPS3k",
        candidates=_IM2GPS_CSV_CANDIDATES,
        output_name="im2gps_italy.csv",
        spatial_filter=spatial_filter,
        image_subdir=image_subdir,
    )


def ensure_yfcc_italy_csv(
    yfcc_root: str,
    spatial_filter: ItalySpatialFilter,
    image_subdir: str = "images",
) -> str:
    """Filter YFCC26k to Italy polygon and cache as yfcc26k_italy.csv.

    Args:
        yfcc_root:     Dir containing yfcc26k_test.csv and images/.
        spatial_filter: Configured ItalySpatialFilter instance.
        image_subdir:  Sub-directory with JPEG files (default: "images").

    Returns:
        Path to the cached Italy CSV.
    """
    return _ensure_italy_csv(
        root=yfcc_root,
        source_name="YFCC26k",
        candidates=_YFCC_CSV_CANDIDATES,
        output_name="yfcc26k_italy.csv",
        spatial_filter=spatial_filter,
        image_subdir=image_subdir,
    )


def describe_italy_subset(csv_path: str, source_name: str = "") -> pd.DataFrame:
    """Print a geographic summary of an Italy-filtered benchmark CSV."""
    df = pd.read_csv(csv_path)

    def _region(lat: float) -> str:
        if lat > 44.0:   return "Nord"
        if lat > 41.5:   return "Centro"
        return "Sud/Isole"

    df["region"] = df["lat"].map(_region)
    label = source_name or os.path.basename(csv_path)
    print(f"\n{label} — {len(df):,} Italy images")
    print(f"  Lat : [{df['lat'].min():.3f}, {df['lat'].max():.3f}]")
    print(f"  Lon : [{df['lon'].min():.3f}, {df['lon'].max():.3f}]")
    print(df["region"].value_counts().to_string())
    return df
