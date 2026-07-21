"""End-to-end benchmark runner."""

from __future__ import annotations

import importlib
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as config_mod
from . import metrics as metrics_mod
from . import predictors as predictors_mod
from .config import BenchmarkConfig, OSV5M_TEST_MIN_TRAIN_SEPARATION_KM
from .datasets import (
    BaseGeolocDataset,
    build_dataloader,
    load_metadata_csv,
    prepare_im2gps_dataset,
    prepare_osv5m_test_dataset,
    prepare_yfcc_dataset,
)
from .metrics import compute_errors_km, compute_metrics, metrics_table, paper_metrics_table
from .predictors import GeolocPredictor, build_predictors
from .spatial import ItalySpatialFilter, verify_min_separation_km
from .visualization import plot_accuracy_bars, plot_error_cdf, plot_worst_cases_map, save_results_bundle


def build_dataset(cfg: BenchmarkConfig, name: str, spatial_filter: ItalySpatialFilter) -> BaseGeolocDataset:
    from .open_domain import ensure_im2gps_italy_csv, ensure_yfcc_italy_csv
    name = name.lower()
    if name == "osv5m":
        return prepare_osv5m_test_dataset(cfg, spatial_filter)
    if name == "im2gps":
        # Auto-create Italy CSV if missing (idempotent). Images must be extracted first
        # (run Section 1 notebook cell once to unzip im2gps3ktest.zip).
        ensure_im2gps_italy_csv(
            cfg.im2gps_root,
            spatial_filter,
            image_subdir=getattr(cfg, "im2gps_imgdir", "im2gps3ktest"),
        )
        return prepare_im2gps_dataset(cfg, spatial_filter)
    if name in ("yfcc", "yfcc26k"):
        ensure_yfcc_italy_csv(
            cfg.yfcc_root,
            spatial_filter,
            image_subdir=getattr(cfg, "yfcc_imgdir", "images"),
        )
        return prepare_yfcc_dataset(cfg, spatial_filter)
    raise ValueError(f"Unknown dataset: {name}")


def verify_osv5m_separation(cfg: BenchmarkConfig) -> dict:
    test_meta = load_metadata_csv(cfg.osv5m_test_meta_csv)
    train_meta = load_metadata_csv(cfg.osv5m_train_meta_csv)
    return verify_min_separation_km(
        test_meta["lat"].values,
        test_meta["lon"].values,
        train_meta["lat"].values,
        train_meta["lon"].values,
        min_km=OSV5M_TEST_MIN_TRAIN_SEPARATION_KM,
    )


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    # Colab caches imported modules — reload so Drive file edits take effect.
    importlib.reload(config_mod)
    importlib.reload(metrics_mod)
    importlib.reload(predictors_mod)
    from .config import BENCHMARK_PACKAGE_VERSION
    from .predictors import PREDICTORS_VERSION, build_predictors

    if BENCHMARK_PACKAGE_VERSION < 4 or PREDICTORS_VERSION < 4:
        print(
            "WARNING: stale benchmarking/predictors.py detected. "
            "Re-sync benchmarking/ to Drive. Applying path-based inference fallback."
        )
        from .predictors import GeolocPredictor, run_path_inference

        def _path_run(self, loader):
            return run_path_inference(self, loader)

        GeolocPredictor.run_on_dataloader = _path_run
        predictors_mod.GeoCLIPRetrievalPredictor.run_on_dataloader = _path_run

    cfg.ensure_output_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    spatial_filter = ItalySpatialFilter(
        use_polygon=cfg.use_polygon_filter,
        poly_buffer_deg=cfg.poly_buffer_deg,
    )

    if "osv5m" in [d.lower() for d in cfg.datasets]:
        try:
            sep = verify_osv5m_separation(cfg)
            print(
                f"OSV5M 1 km separation check (sample n={sep['checked']}): "
                f"violations={sep['violations']}, min observed={sep['min_observed_km']:.2f} km"
            )
        except Exception as exc:
            print(f"WARNING: OSV5M separation check skipped ({type(exc).__name__}: {exc})")

    # Build predictors — individual failures are caught inside build_predictors already,
    # but wrap the whole call so a catastrophic error doesn't kill the run.
    try:
        predictors = build_predictors(cfg, device)
    except Exception as exc:
        print(f"FATAL: build_predictors failed: {exc}")
        raise

    all_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    skipped_datasets: List[str] = []

    for dataset_name in cfg.datasets:
        print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")

        # ── Dataset loading ────────────────────────────────────────────────────
        try:
            dataset = build_dataset(cfg, dataset_name, spatial_filter)
        except Exception as exc:
            print(
                f"\n  SKIPPED dataset '{dataset_name}': {type(exc).__name__}: {exc}\n"
                f"  Fix the issue (see message above) and re-run — other datasets continue."
            )
            skipped_datasets.append(dataset_name)
            continue

        loader = build_dataloader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        print(f"Samples: {len(dataset):,}")

        # ── Per-model inference ────────────────────────────────────────────────
        dataset_results: Dict[str, pd.DataFrame] = {}
        for predictor in predictors:
            try:
                rows = predictor.run_on_dataloader(loader)
                df = pd.DataFrame(rows).dropna(subset=["lat_true", "lon_true"])
                valid = compute_errors_km(df) if not df.empty else df
                n_total = len(df)
                n_valid = len(valid)
                if n_valid == 0:
                    print(
                        f"\n  WARNING: {predictor.name} produced 0/{n_total} valid predictions."
                    )
                else:
                    df = valid
                dataset_results[predictor.name] = df
                m = compute_metrics(df)
                m.pretty_print(predictor.name)
                m.pretty_print(predictor.name, paper_format=True)
            except Exception as exc:
                print(f"\n  SKIPPED model '{predictor.name}' on '{dataset_name}': {type(exc).__name__}: {exc}")
                dataset_results[predictor.name] = pd.DataFrame()

        all_results[dataset_name] = dataset_results

        # ── Save + plots ───────────────────────────────────────────────────────
        try:
            save_results_bundle(dataset_results, cfg.output_dir, dataset_name)
        except Exception as exc:
            print(f"  WARNING: save_results_bundle failed: {exc}")

        try:
            paper_df = paper_metrics_table(dataset_results)
            paper_path = f"{cfg.output_dir}/{dataset_name}_paper_metrics.csv"
            paper_df.to_csv(paper_path, index=False)
            print(f"\nGeoCLIP paper-format comparison ({dataset_name}):")
            print(paper_df.to_string(index=False))
            print(f"Saved: {paper_path}")
        except Exception as exc:
            print(f"  WARNING: metrics table failed: {exc}")

        plot_results = {k: v for k, v in dataset_results.items() if not v.empty}
        ft_name = predictors[0].name
        base_name = "geoclip_retrieval_italy"
        if base_name not in plot_results:
            base_name = next(
                (p.name for p in predictors[1:] if p.name in plot_results), None
            )

        if plot_results:
            try:
                plot_error_cdf(
                    plot_results,
                    output_path=f"{cfg.output_dir}/{dataset_name}_error_cdf.png",
                )
                plot_accuracy_bars(
                    plot_results,
                    output_path=f"{cfg.output_dir}/{dataset_name}_accuracy_bars.png",
                )
            except Exception as exc:
                print(f"  WARNING: plot generation failed: {exc}")

        if ft_name in plot_results and not compute_errors_km(plot_results[ft_name]).empty:
            try:
                plot_worst_cases_map(
                    plot_results[ft_name],
                    plot_results.get(base_name) if base_name else None,
                    n_worst=20,
                    output_path=f"{cfg.output_dir}/{dataset_name}_worst20_map.html",
                    reference_label="Fine-tuned",
                    compare_label="GeoCLIP retrieval",
                )
            except Exception as exc:
                print(f"  WARNING: worst-case map failed: {exc}")

        try:
            print("\nSummary table:")
            print(metrics_table(dataset_results).to_string(index=False))
        except Exception as exc:
            print(f"  WARNING: summary table failed: {exc}")

    if skipped_datasets:
        print(f"\n{'=' * 60}")
        print(f"Skipped datasets (errors above): {skipped_datasets}")
        print("Completed datasets:", list(all_results.keys()))

    return all_results


def regenerate_plots(cfg, results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Re-generate plots/maps from saved results without re-running inference."""

    cfg.ensure_output_dir()
    for dataset_name, dataset_results in results.items():
        plot_results = {k: v for k, v in dataset_results.items() if not v.empty}
        if plot_results:
            plot_error_cdf(
                plot_results,
                output_path=f"{cfg.output_dir}/{dataset_name}_error_cdf.png",
            )
            plot_accuracy_bars(
                plot_results,
                output_path=f"{cfg.output_dir}/{dataset_name}_accuracy_bars.png",
            )
        ft_name = "geoclip_italy_finetuned"
        base_name = "geoclip_retrieval_italy"
        if ft_name in plot_results:
            plot_worst_cases_map(
                plot_results[ft_name],
                plot_results.get(base_name),
                n_worst=20,
                output_path=f"{cfg.output_dir}/{dataset_name}_worst20_map.html",
                reference_label="Fine-tuned",
                compare_label="GeoCLIP retrieval (Italy)",
            )
