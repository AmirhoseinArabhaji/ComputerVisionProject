"""Spatial filtering utilities for Italy-constrained evaluation."""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import MAX_LAT, MAX_LON, MIN_LAT, MIN_LON

LatLon = Tuple[float, float]

_SAMPLE_POLYGON: Sequence[LatLon] = (
    (47.1, 6.6), (47.1, 18.5), (45.0, 18.5), (43.5, 15.0),
    (41.0, 15.0), (38.0, 18.0), (38.0, 14.0), (36.0, 14.0), (36.0, 6.6),
)


def _ray_cast(lat: float, lon: float, polygon: Sequence[LatLon]) -> bool:
    inside, x, y, n = False, lon, lat, len(polygon)
    for i in range(n):
        j = (i - 1) % n
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((xi > x) != (xj > x)) and (y < (yj - yi) * (x - xi) / (xj - xi + 1e-12) + yi):
            inside = not inside
    return inside


class ItalySpatialFilter:
    """Point-in-Italy tester using polygon (preferred) or bounding box."""

    def __init__(self, use_polygon: bool = True, poly_buffer_deg: float = 0.02) -> None:
        self.use_polygon = use_polygon
        self.poly_buffer_deg = poly_buffer_deg
        self._prepared_geom = None
        self._fallback_polygon: Optional[Sequence[LatLon]] = None
        if use_polygon:
            self._load_polygon()

    def _load_polygon(self) -> None:
        try:
            import geopandas as gpd
            from shapely.prepared import prep

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
                except Exception:
                    world = gpd.read_file(
                        "https://naturalearth.s3.amazonaws.com/110m_cultural/"
                        "ne_110m_admin_0_countries.zip"
                    )
            name_col = next(
                (c for c in world.columns if c.upper() == "NAME" and "LONG" not in c.upper()),
                "NAME",
            )
            geom = world[world[name_col] == "Italy"].geometry.iloc[0]
            if self.poly_buffer_deg > 0:
                geom = geom.buffer(self.poly_buffer_deg)
            self._prepared_geom = prep(geom)
        except Exception as exc:
            warnings.warn(f"Polygon filter unavailable ({exc}); using ray-casting fallback.")
            self._fallback_polygon = _SAMPLE_POLYGON

    def in_bbox(self, lat: float, lon: float) -> bool:
        return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON

    def contains(self, lat: float, lon: float) -> bool:
        if not self.in_bbox(lat, lon):
            return False
        if not self.use_polygon:
            return True
        if self._prepared_geom is not None:
            from shapely.geometry import Point

            return bool(self._prepared_geom.contains(Point(lon, lat)))
        if self._fallback_polygon is not None:
            return _ray_cast(lat, lon, self._fallback_polygon)
        return True

    def mask_array(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        bbox = (
            (lats >= MIN_LAT) & (lats <= MAX_LAT) &
            (lons >= MIN_LON) & (lons <= MAX_LON)
        )
        if not self.use_polygon or (self._prepared_geom is None and self._fallback_polygon is None):
            return bbox
        out = np.zeros(len(lats), dtype=bool)
        idx = np.where(bbox)[0]
        for i in idx:
            out[i] = self.contains(float(lats[i]), float(lons[i]))
        return out

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        desc: str = "Italy filter",
    ) -> pd.DataFrame:
        work = df.copy()
        if lat_col not in work.columns:
            for alt in ("latitude", "Latitude", "LAT"):
                if alt in work.columns:
                    lat_col = alt
                    break
        if lon_col not in work.columns:
            for alt in ("longitude", "Longitude", "LON"):
                if alt in work.columns:
                    lon_col = alt
                    break

        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
        work = work.dropna(subset=[lat_col, lon_col])

        if self.use_polygon and self._prepared_geom is not None:
            from shapely.geometry import Point

            lats = work[lat_col].astype(float).values
            lons = work[lon_col].astype(float).values
            mask = [
                self.in_bbox(lat, lon) and self._prepared_geom.contains(Point(lon, lat))
                for lat, lon in tqdm(zip(lats, lons), total=len(work), desc=desc, unit="rows")
            ]
        elif self.use_polygon and self._fallback_polygon is not None:
            mask = work.apply(
                lambda r: self.contains(float(r[lat_col]), float(r[lon_col])),
                axis=1,
            )
        else:
            mask = (
                (work[lat_col] >= MIN_LAT) & (work[lat_col] <= MAX_LAT) &
                (work[lon_col] >= MIN_LON) & (work[lon_col] <= MAX_LON)
            )
        return work.loc[mask].reset_index(drop=True)

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rename = {}
        if "image_id" not in out.columns:
            for alt in ("id", "photo_id", "img_id", "filename"):
                if alt in out.columns:
                    rename[alt] = "image_id"
                    break
        if "lat" not in out.columns:
            for alt in ("latitude", "Latitude"):
                if alt in out.columns:
                    rename[alt] = "lat"
                    break
        if "lon" not in out.columns:
            for alt in ("longitude", "Longitude", "lon"):
                if alt in out.columns:
                    rename[alt] = "lon"
                    break
        if rename:
            out = out.rename(columns=rename)
        return out


def haversine_km_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km (vectorized, WGS84 sphere)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def verify_min_separation_km(
    test_lats: np.ndarray,
    test_lons: np.ndarray,
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
    min_km: float,
    sample_size: int = 500,
) -> dict:
    """Check OSV5M-style 1 km train/test separation on a sample."""
    n = min(sample_size, len(test_lats))
    if n == 0:
        return {"checked": 0, "violations": 0, "min_observed_km": float("nan")}
    idx = np.linspace(0, len(test_lats) - 1, n, dtype=int)
    mins = []
    violations = 0
    for i in idx:
        d = haversine_km_vectorized(
            np.full(len(ref_lats), test_lats[i]),
            np.full(len(ref_lons), test_lons[i]),
            ref_lats,
            ref_lons,
        )
        m = float(np.min(d))
        mins.append(m)
        if m < min_km:
            violations += 1
    return {
        "checked": n,
        "violations": violations,
        "min_observed_km": float(np.min(mins)),
        "median_min_km": float(np.median(mins)),
    }
