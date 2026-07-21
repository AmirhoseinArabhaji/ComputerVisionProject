"""Evaluation metrics for geolocalization benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from . import config as _benchmark_config
from .spatial import haversine_km_vectorized

# Fallbacks when Drive has a partially synced config.py (older copy).
ACCURACY_THRESHOLDS_KM = getattr(
    _benchmark_config, "ACCURACY_THRESHOLDS_KM", (1, 10, 25, 50, 100, 200, 500, 2500)
)
PAPER_THRESHOLDS_KM = getattr(_benchmark_config, "PAPER_THRESHOLDS_KM", (1, 25, 200, 2500))


@dataclass
class GeolocMetrics:
    n: int
    mean_km: float
    median_km: float
    p90_km: float
    accuracy_at_km: Dict[int, float]

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        out = {
            f"{prefix}n": self.n,
            f"{prefix}mean_km": self.mean_km,
            f"{prefix}median_km": self.median_km,
            f"{prefix}p90_km": self.p90_km,
        }
        for th, pct in self.accuracy_at_km.items():
            out[f"{prefix}acc_{th}km_pct"] = pct
        return out

    def pretty_print(self, label: str, paper_format: bool = False) -> None:
        print(f"\n  {label}")
        print(f"    N            : {self.n:,}")
        if self.n == 0:
            print("    No valid predictions (all inference failed or missing coordinates).")
            return
        print(f"    Mean error   : {self.mean_km:.2f} km")
        print(f"    Median error : {self.median_km:.2f} km")
        if not paper_format:
            print(f"    90th pct     : {self.p90_km:.2f} km")
        thresholds = PAPER_THRESHOLDS_KM if paper_format else self.accuracy_at_km.keys()
        header = "GeoCLIP paper format" if paper_format else "Accuracy @ threshold"
        print(f"    --- {header} ---")
        for th in thresholds:
            pct = self.accuracy_at_km.get(int(th), 0.0)
            print(f"    Within {th:>5} km : {pct:.1f}%")


def compute_errors_km(df: pd.DataFrame) -> pd.DataFrame:
    """Add error_km column; drop rows with missing predictions or coordinates."""
    required = {"lat_true", "lon_true", "lat_pred", "lon_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results DataFrame missing columns: {missing}")

    out = df.dropna(subset=list(required)).copy()
    if out.empty:
        return out
    out["error_km"] = haversine_km_vectorized(
        out["lat_true"].values,
        out["lon_true"].values,
        out["lat_pred"].values,
        out["lon_pred"].values,
    )
    return out


def compute_metrics(
    df: pd.DataFrame,
    thresholds_km: Sequence[int] = ACCURACY_THRESHOLDS_KM,
) -> GeolocMetrics:
    empty_acc = {int(th): 0.0 for th in thresholds_km}
    if df.empty:
        return GeolocMetrics(0, float("nan"), float("nan"), float("nan"), empty_acc)

    work = compute_errors_km(df) if "error_km" not in df.columns else df.dropna(subset=["error_km"])
    if work.empty:
        return GeolocMetrics(0, float("nan"), float("nan"), float("nan"), empty_acc)

    e = work["error_km"].values
    acc = {int(th): float(np.mean(e <= th) * 100.0) for th in thresholds_km}
    return GeolocMetrics(
        n=len(work),
        mean_km=float(np.mean(e)),
        median_km=float(np.median(e)),
        p90_km=float(np.percentile(e, 90)),
        accuracy_at_km=acc,
    )


def metrics_table(
    results: Dict[str, pd.DataFrame],
    model_names: Iterable[str] | None = None,
    thresholds_km: Sequence[int] = ACCURACY_THRESHOLDS_KM,
) -> pd.DataFrame:
    """Build a comparison table across models."""
    rows = []
    names = list(model_names) if model_names is not None else list(results.keys())
    for name in names:
        if name not in results:
            continue
        m = compute_metrics(results[name], thresholds_km=thresholds_km)
        row = {"model": name, **m.to_dict()}
        rows.append(row)
    return pd.DataFrame(rows)


def paper_metrics_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    GeoCLIP paper-style table: mean/median error + acc @ 1/25/200/2500 km.

    Comparable to Table 1 in the GeoCLIP NeurIPS 2023 paper (Im2GPS3k / YFCC26k
    protocol, applied here to the Italy test subset).
    """
    rows = []
    for name, df in results.items():
        m = compute_metrics(df, thresholds_km=PAPER_THRESHOLDS_KM)
        rows.append({
            "model": name,
            "mean_km": round(m.mean_km, 1) if m.n > 0 else None,
            "median_km": round(m.median_km, 1) if m.n > 0 else None,
            "acc@1km": round(m.accuracy_at_km[1], 1) if m.n > 0 else 0.0,
            "acc@25km": round(m.accuracy_at_km[25], 1) if m.n > 0 else 0.0,
            "acc@200km": round(m.accuracy_at_km[200], 1) if m.n > 0 else 0.0,
            "acc@2500km": round(m.accuracy_at_km[2500], 1) if m.n > 0 else 0.0,
            "n": m.n,
        })
    return pd.DataFrame(rows)
