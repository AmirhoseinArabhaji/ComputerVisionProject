"""Unified geolocalization predictor interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import MAX_LAT, MAX_LON, MIN_LAT, MIN_LON
from .datasets import build_tta_transforms, default_eval_transform
from .models import GeoClipItaly, load_finetuned_checkpoint, patch_geoclip_image_encoder
from .spatial import ItalySpatialFilter

# Bump when inference loop semantics change (used by notebook sanity check).
PREDICTORS_VERSION = 5


def denorm(norm_lat: float, norm_lon: float) -> Tuple[float, float]:
    lat = norm_lat * (MAX_LAT - MIN_LAT) + MIN_LAT
    lon = norm_lon * (MAX_LON - MIN_LON) + MIN_LON
    return float(lat), float(lon)


def _preprocess_geoclip_image(geoclip_model, image_path: str, device: torch.device) -> torch.Tensor:
    """Load image and return a (1, 3, H, W) tensor — no double batching."""
    img = Image.open(image_path).convert("RGB")
    pixel = geoclip_model.image_encoder.preprocess_image(img).to(device)
    if pixel.dim() == 3:
        pixel = pixel.unsqueeze(0)
    return pixel


@torch.no_grad()
def geoclip_retrieve_from_path(
    geoclip_model,
    image_path: str,
    gallery: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Image-to-GPS retrieval mirroring GeoCLIP.forward() internals.

    Uses image_encoder -> location_encoder cosine similarity over `gallery`.
    """
    pixel = _preprocess_geoclip_image(geoclip_model, image_path, device)
    image_features = geoclip_model.image_encoder(pixel)
    image_features = F.normalize(image_features, dim=1)

    gallery = gallery.to(device)
    location_features = geoclip_model.location_encoder(gallery)
    location_features = F.normalize(location_features, dim=1)

    logit_scale = geoclip_model.logit_scale.exp()
    logits = logit_scale * (image_features @ location_features.t())
    best_idx = int(logits[0].argmax().item())
    gps = gallery[best_idx].detach().cpu()
    return float(gps[0].item()), float(gps[1].item())


def run_path_inference(predictor: "GeolocPredictor", loader: DataLoader) -> List[dict]:
    """Shared path-based eval loop for retrieval baselines."""
    rows: List[dict] = []
    for _imgs, _targets, metas in tqdm(loader, desc=predictor.name, leave=False):
        for meta in metas:
            lat, lon = predictor.predict_path(meta["image_path"])
            rows.append(GeolocPredictor._row(meta, lat, lon))
    return rows


class GeolocPredictor(ABC):
    """Common interface for all geolocalization models."""

    name: str = "predictor"

    @abstractmethod
    def predict_path(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        ...

    def predict_batch_paths(self, paths: Sequence[str]) -> List[Tuple[Optional[float], Optional[float]]]:
        return [self.predict_path(p) for p in paths]

    def run_on_dataloader(self, loader: DataLoader) -> List[dict]:
        return run_path_inference(self, loader)

    @staticmethod
    def _row(meta: dict, lat: Optional[float], lon: Optional[float]) -> dict:
        return {
            "image_id": meta.get("image_id", ""),
            "image_path": meta.get("image_path", ""),
            "lat_true": meta.get("lat"),
            "lon_true": meta.get("lon"),
            "lat_pred": lat,
            "lon_pred": lon,
            "source": meta.get("source", ""),
        }


class GeoClipItalyPredictor(GeolocPredictor):
    """Fine-tuned regression model with optional TTA."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        lora_rank: int = 16,
        lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
        use_tta: bool = True,
        n_tta_views: int = 6,
    ) -> None:
        self.name = "geoclip_italy_finetuned"
        self.device = device
        self.use_tta = use_tta
        self.n_tta_views = n_tta_views
        self.eval_transform = default_eval_transform()
        self.tta_transforms = build_tta_transforms()
        self.model = GeoClipItaly(
            lora_rank=lora_rank,
            target_modules=list(lora_target_modules),
        ).to(device)
        meta = load_finetuned_checkpoint(self.model, checkpoint_path, device)
        self.checkpoint_meta = meta
        print(f"Loaded fine-tuned checkpoint: {checkpoint_path}")

    def _predict_tensor(self, tensor: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
        try:
            with torch.no_grad():
                out = self.model(tensor.to(self.device)).cpu().numpy()[0]
            return denorm(float(out[0]), float(out[1]))
        except Exception:
            return None, None

    def predict_path(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        if not image_path or not os.path.exists(image_path):
            return None, None
        try:
            img = Image.open(image_path).convert("RGB")
            if self.use_tta:
                lats, lons = [], []
                for tf in self.tta_transforms[: self.n_tta_views]:
                    tensor = tf(img).unsqueeze(0)
                    lat, lon = self._predict_tensor(tensor)
                    if lat is not None:
                        lats.append(lat)
                        lons.append(lon)
                if not lats:
                    return None, None
                return float(np.mean(lats)), float(np.mean(lons))
            tensor = self.eval_transform(img).unsqueeze(0)
            return self._predict_tensor(tensor)
        except Exception:
            return None, None

    def run_on_dataloader(self, loader: DataLoader) -> List[dict]:
        """TTA uses path-based views; otherwise batched GPU inference."""
        if self.use_tta:
            return super().run_on_dataloader(loader)
        rows: List[dict] = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _targets, metas in tqdm(loader, desc=self.name, leave=False):
                if imgs is None:
                    for meta in metas:
                        lat, lon = self.predict_path(meta["image_path"])
                        rows.append(self._row(meta, lat, lon))
                    continue
                out = self.model(imgs.to(self.device)).cpu().numpy()
                for i, meta in enumerate(metas):
                    lat, lon = denorm(float(out[i, 0]), float(out[i, 1]))
                    rows.append(self._row(meta, lat, lon))
        return rows


class GeoCLIPRetrievalPredictor(GeolocPredictor):
    """
    Pretrained GeoCLIP image-to-GPS retrieval baseline (paper method).

    When italy_constrained=True, retrieval is restricted to GPS gallery cells
    inside the Italy polygon. When False, uses the full 100K global gallery
    (same setting as the GeoCLIP paper evaluation).
    """

    def __init__(self, device: torch.device, italy_constrained: bool = True) -> None:
        from geoclip.model import GeoCLIP

        self.name = (
            "geoclip_retrieval_italy" if italy_constrained else "geoclip_retrieval_global"
        )
        self.device = device
        self.italy_constrained = italy_constrained
        self._geoclip = GeoCLIP(from_pretrained=True)
        patch_geoclip_image_encoder(self._geoclip)
        self._geoclip.eval().to(device)

        self._gallery = self._geoclip.gps_gallery
        if italy_constrained:
            spatial = ItalySpatialFilter(use_polygon=True)
            lats = self._gallery[:, 0].cpu().numpy()
            lons = self._gallery[:, 1].cpu().numpy()
            mask = spatial.mask_array(lats, lons)
            if not np.any(mask):
                raise RuntimeError("Italy gallery mask is empty.")
            self._gallery = self._gallery[torch.from_numpy(mask)]
            print(
                f"Italy-constrained gallery: {self._gallery.shape[0]:,} / "
                f"{self._geoclip.gps_gallery.shape[0]:,} GPS cells"
            )
        else:
            print(f"Global gallery: {self._gallery.shape[0]:,} GPS cells")

    def predict_path(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        if not image_path or not os.path.exists(image_path):
            return None, None
        try:
            lat, lon = geoclip_retrieve_from_path(
                self._geoclip, image_path, self._gallery, self.device
            )
            return lat, lon
        except Exception as exc:
            if not getattr(self, "_warned", False):
                print(f"WARNING: {self.name} inference failed ({image_path}): {exc}")
                self._warned = True
            return None, None


class GeoEstimationPredictor(GeolocPredictor):
    """
    TIBHannover GeoEstimation — open-source ISNs equivalent used in the GeoCLIP paper.

    EfficientNet-B4 backbone trained on MP-16 (4.72M geotagged Flickr images, same
    training set as GeoCLIP).  Hierarchical S2-cell classification (street / city /
    region / continent scales).

    Italy-constrained mode: introspects the estimator to discover which S2 class indices
    fall inside Italy.  At inference it filters the top-K predictions to Italy-only classes
    and returns the highest-scoring one.

    Requirements:
        pip install git+https://github.com/TIBHannover/GeoEstimation.git

    Reference:
        Müller & Riedl, "Geolocation Estimation of Photos using a Hierarchical Model and
        Scene Classification", ECCV 2018.  The NeurIPS 2022 update (EfficientNet-B4) is
        referenced as ISNs [12] in the GeoCLIP paper.
    """

    def __init__(
        self,
        device: torch.device,
        spatial_filter: ItalySpatialFilter,
        italy_constrained: bool = True,
    ) -> None:
        self.device = device
        self.italy_constrained = italy_constrained
        self.name = "geo_estimation_italy" if italy_constrained else "geo_estimation_global"
        self._warned = False
        self._sf = spatial_filter
        self._italy_classes: Optional[set] = None

        try:
            from geo_estimation import GeoEstimator  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "geo_estimation package not found.\n"
                "Add this line to the install cell in the notebook:\n"
                "  %pip install git+https://github.com/TIBHannover/GeoEstimation.git\n"
                "then Runtime → Restart session and re-run."
            ) from exc

        self._estimator = GeoEstimator()

        if italy_constrained:
            self._italy_classes = self._discover_italy_classes()
            n = len(self._italy_classes) if self._italy_classes else "unknown"
            print(f"GeoEstimation (Italy-constrained): {n} Italy S2 classes")
        else:
            print("GeoEstimation loaded (global, unconstrained).")

    # ------------------------------------------------------------------
    # Italy S2-cell discovery
    # ------------------------------------------------------------------

    def _discover_italy_classes(self) -> Optional[set]:
        """
        Introspect the GeoEstimator for cell GPS centres and return the set of
        class indices whose centre lies inside the Italy polygon.
        """
        est = self._estimator

        # Strategy A: try common DataFrame meta attributes
        for meta_attr in ("meta", "cell_meta", "partitioning", "classes", "hierarchy"):
            meta = getattr(est, meta_attr, None)
            if meta is None:
                continue
            cols = getattr(meta, "columns", [])
            lat_col = next((c for c in ("lat_mean", "latitude_mean", "lat", "latitude") if c in cols), None)
            lon_col = next((c for c in ("lon_mean", "longitude_mean", "lon", "longitude") if c in cols), None)
            if lat_col and lon_col:
                mask = self._sf.mask_array(
                    meta[lat_col].values.astype(float),
                    meta[lon_col].values.astype(float),
                )
                return set(int(i) for i in np.where(mask)[0])

        # Strategy B: try GPS array attributes
        for gps_attr in ("gps_cells", "cell_gps", "gps_gallery", "cell_locations"):
            arr = getattr(est, gps_attr, None)
            if arr is not None:
                try:
                    lats = np.array([float(g[0]) for g in arr])
                    lons = np.array([float(g[1]) for g in arr])
                    mask = self._sf.mask_array(lats, lons)
                    return set(int(i) for i in np.where(mask)[0])
                except Exception:
                    continue

        import warnings
        warnings.warn(
            f"[{self.name}] Cannot introspect S2 cell GPS — "
            "running unconstrained (predictions may fall outside Italy)."
        )
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_path(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        if not image_path or not os.path.exists(image_path):
            return None, None
        try:
            # Try top-K API first so we can filter to Italy classes
            if self.italy_constrained and self._italy_classes:
                for k in (20, 100):
                    try:
                        topk = self._estimator.get_prediction(image_path, top_k=k)
                    except TypeError:
                        break  # top_k kwarg not supported — fall through
                    if isinstance(topk, (list, tuple)):
                        for pred in topk:
                            idx = (pred.get("class_idx") if isinstance(pred, dict)
                                   else (pred[0] if len(pred) > 2 else None))
                            if idx is not None and int(idx) in self._italy_classes:
                                return self._extract_latlon(pred)
                    break

            # Standard single-prediction
            result = self._estimator.get_prediction(image_path)
            return self._extract_latlon(result)

        except Exception as exc:
            if not self._warned:
                print(f"WARNING [{self.name}]: inference failed: {exc}")
                self._warned = True
            return None, None

    @staticmethod
    def _extract_latlon(result) -> Tuple[Optional[float], Optional[float]]:
        if result is None:
            return None, None
        if isinstance(result, dict):
            lat = result.get("lat") or result.get("latitude") or result.get("pred_lat")
            lon = result.get("lon") or result.get("longitude") or result.get("pred_lon")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            # (lat, lon) or (class_idx, lat, lon, ...)
            if len(result) >= 3 and isinstance(result[0], int):
                return float(result[1]), float(result[2])
            return float(result[0]), float(result[1])
        return None, None


class StreetCLIPPredictor(GeolocPredictor):
    """
    StreetCLIP (Haas et al., "Learning to Geolocalize with CLIP", 2023).
    CLIP ViT-L/14 fine-tuned on geolocalized Flickr images.
    HuggingFace model ID: geolocal/StreetCLIP

    At inference: image → StreetCLIP image features → cosine similarity over a
    text-encoded GPS gallery → best-matching GPS returned.

    Gallery: loaded from OSV5M train metadata (real road/settlement locations) when
    available; falls back to random samples from the Italy polygon.

    No extra install required — transformers is a transitive dependency of geoclip.
    """

    name = "streetclip_italy"
    _MODEL_ID = "geolocal/StreetCLIP"
    _TEXT_TEMPLATE = "A photo taken in Italy at latitude {lat:.4f}, longitude {lon:.4f}"

    def __init__(
        self,
        device: torch.device,
        spatial_filter: ItalySpatialFilter,
        train_meta_csv: Optional[str] = None,
        gallery_size: int = 50_000,
    ) -> None:
        self.device = device
        self._warned = False

        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers not found — run: pip install transformers"
            ) from exc

        print(f"Loading {self._MODEL_ID} ...")
        self._model = CLIPModel.from_pretrained(self._MODEL_ID).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(self._MODEL_ID)

        self._gallery_gps, self._text_features = self._build_gallery(
            spatial_filter, train_meta_csv, gallery_size
        )
        print(f"StreetCLIP gallery: {len(self._gallery_gps):,} Italy GPS points")

    # ------------------------------------------------------------------
    # Gallery construction
    # ------------------------------------------------------------------

    def _build_gallery(
        self,
        spatial_filter: ItalySpatialFilter,
        train_meta_csv: Optional[str],
        n: int,
    ) -> Tuple[list, torch.Tensor]:
        gps_pts: list = []

        # Prefer OSV5M train metadata — real on-road/settlement locations
        if train_meta_csv and os.path.exists(train_meta_csv):
            try:
                df = pd.read_csv(train_meta_csv, usecols=["latitude", "longitude"]).dropna()
                mask = spatial_filter.mask_array(df["latitude"].values, df["longitude"].values)
                df = df[mask]
                if len(df) > n:
                    df = df.sample(n=n, random_state=42)
                gps_pts = list(zip(df["latitude"].tolist(), df["longitude"].tolist()))
                print(f"  Gallery source : OSV5M train metadata ({len(gps_pts):,} pts)")
            except Exception as exc:
                print(f"  WARNING: Could not load train metadata ({exc}). Using random fallback.")

        if not gps_pts:
            rng = np.random.default_rng(42)
            while len(gps_pts) < n:
                lats = rng.uniform(MIN_LAT, MAX_LAT, 5_000)
                lons = rng.uniform(MIN_LON, MAX_LON, 5_000)
                mask = spatial_filter.mask_array(lats, lons)
                for lat, lon in zip(lats[mask], lons[mask]):
                    gps_pts.append((float(lat), float(lon)))
                    if len(gps_pts) >= n:
                        break
            print(f"  Gallery source : random Italy polygon ({len(gps_pts):,} pts)")

        texts = [self._TEXT_TEMPLATE.format(lat=lat, lon=lon) for lat, lon in gps_pts]
        feat_chunks: List[torch.Tensor] = []
        bs = 256
        print(f"  Encoding {len(texts):,} text descriptions ...")
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), bs), desc="text encode", leave=False):
                enc = self._processor(
                    text=texts[i : i + bs],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self._model.get_text_features(**enc)
                if not isinstance(out, torch.Tensor):
                    out = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0, :]
                feat_chunks.append(F.normalize(out, dim=-1).cpu())

        return gps_pts, torch.cat(feat_chunks, dim=0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_path(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        if not image_path or not os.path.exists(image_path):
            return None, None
        try:
            img = Image.open(image_path).convert("RGB")
            enc = self._processor(images=img, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                img_feat = self._model.get_image_features(**enc)
                if not isinstance(img_feat, torch.Tensor):
                    img_feat = img_feat.pooler_output if hasattr(img_feat, "pooler_output") else img_feat.last_hidden_state[:, 0, :]
                img_feat = F.normalize(img_feat, dim=-1).cpu()
            best_idx = int((img_feat @ self._text_features.t())[0].argmax())
            lat, lon = self._gallery_gps[best_idx]
            return float(lat), float(lon)
        except Exception as exc:
            if not self._warned:
                print(f"WARNING [streetclip_italy]: {exc}")
                self._warned = True
            return None, None


def build_predictors(cfg, device: torch.device) -> List[GeolocPredictor]:
    """Build the ordered list of predictors for a benchmark run.

    Always includes:
      1. GeoClipItaly (fine-tuned regression)
      2. GeoCLIP retrieval — Italy-constrained gallery (base-model comparison)

    Optional (set flags in BenchmarkConfig / notebook config cell):
      3. GeoEstimation (ISNs equivalent) — if cfg.include_geoestimation = True
      4. GeoCLIP retrieval — global gallery (ablation) — if cfg.include_global_geoclip_baseline = True
    """
    predictors: List[GeolocPredictor] = [
        GeoClipItalyPredictor(
            checkpoint_path=cfg.checkpoint_path,
            device=device,
            lora_rank=cfg.lora_rank,
            lora_target_modules=cfg.lora_target_modules,
            use_tta=cfg.use_tta,
            n_tta_views=cfg.n_tta_views,
        ),
        GeoCLIPRetrievalPredictor(device=device, italy_constrained=True),
    ]

    if getattr(cfg, "include_geoestimation", False):
        try:
            sf = ItalySpatialFilter(use_polygon=getattr(cfg, "use_polygon_filter", True))
            predictors.append(GeoEstimationPredictor(device=device, spatial_filter=sf))
        except Exception as e:
            print(f"WARNING: GeoEstimation skipped — {type(e).__name__}: {e}")

    if getattr(cfg, "include_streetclip", False):
        try:
            sf = ItalySpatialFilter(use_polygon=getattr(cfg, "use_polygon_filter", True))
            train_csv = getattr(cfg, "osv5m_train_meta_csv", None)
            predictors.append(
                StreetCLIPPredictor(device=device, spatial_filter=sf, train_meta_csv=train_csv)
            )
        except Exception as e:
            print(f"WARNING: StreetCLIP skipped — {type(e).__name__}: {e}")

    # Ablation only: global (unconstrained) GeoCLIP gallery
    if getattr(cfg, "include_global_geoclip_baseline", False):
        predictors.append(GeoCLIPRetrievalPredictor(device=device, italy_constrained=False))

    return predictors
