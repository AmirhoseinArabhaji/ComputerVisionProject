"""Visualization and error analysis for geolocalization benchmarks."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ACCURACY_THRESHOLDS_KM
from .metrics import PAPER_THRESHOLDS_KM, compute_errors_km, compute_metrics


def plot_error_cdf(
    results: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    max_km: float = 2000.0,
    title: str = "Geolocalization Error CDF (Italy)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("husl", n_colors=len(results))
    for (name, df), color in zip(results.items(), palette):
        work = compute_errors_km(df)
        errors = np.sort(work["error_km"].values)
        if len(errors) == 0:
            continue
        y = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, y, label=name, color=color, linewidth=2)
    ax.set_xlim(0, max_km)
    ax.set_xlabel("Haversine error (km)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved CDF: {output_path}")
    return fig


def plot_accuracy_bars(
    results: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    thresholds_km: Sequence[int] = PAPER_THRESHOLDS_KM,
    title: str = "Accuracy @ Distance Threshold (GeoCLIP paper format)",
) -> plt.Figure:
    models = [m for m, df in results.items() if not df.empty]
    if not models:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No valid predictions to plot", ha="center", va="center")
        ax.axis("off")
        return fig
    x = np.arange(len(thresholds_km))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(models):
        m = compute_metrics(results[name], thresholds_km=thresholds_km)
        vals = [m.accuracy_at_km[int(t)] for t in thresholds_km]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} km" for t in thresholds_km])
    ax.set_ylabel("% within threshold")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_scatter(
    results: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    keys: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    panel_size: float = 5.0,
) -> plt.Figure:
    """Ground-truth vs. predicted GPS scatter, one panel per model.

    `labels` must map every plotted result key (e.g. "geoclip_retrieval_global") to the
    exact caption its panel should show (e.g. "GeoCLIP -- global gallery"). This is
    mandatory rather than falling back to the raw key, so a reader can never end up
    looking at an unconstrained-gallery baseline captioned just "Baseline GeoCLIP".
    """
    from .config import MAX_LAT, MAX_LON, MIN_LAT, MIN_LON

    names = [k for k in (keys or results.keys()) if k in results and not results[k].empty]
    if not names:
        raise ValueError("No non-empty results to plot.")
    missing = [n for n in names if n not in labels]
    if missing:
        raise ValueError(f"`labels` is missing an explicit caption for: {missing}")

    fig, axes = plt.subplots(1, len(names), figsize=(panel_size * len(names), panel_size + 1), squeeze=False)
    palette = sns.color_palette("husl", n_colors=len(names))

    for ax, name, color in zip(axes[0], names, palette):
        df = compute_errors_km(results[name])
        ax.scatter(df["lon_true"], df["lat_true"], c="green", s=10, alpha=0.5, label="Ground truth")
        ax.scatter(df["lon_pred"], df["lat_pred"], c=color, s=10, alpha=0.5, label="Predicted")
        ax.set_xlim(MIN_LON - 0.5, MAX_LON + 0.5)
        ax.set_ylim(MIN_LAT - 0.5, MAX_LAT + 0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(labels[name])
        ax.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved scatter: {output_path}")
    return fig


def plot_region_confusion(
    results: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    keys: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Macro-region (Nord / Centro / Sud-Isole) confusion matrix, one panel per model.

    Same `labels`-is-mandatory contract as `plot_prediction_scatter` -- every panel
    must be captioned with which model/baseline produced it.
    """
    from sklearn.metrics import confusion_matrix

    def macro_region(lat: float) -> str:
        if lat > 44.0:
            return "Nord"
        if lat > 41.5:
            return "Centro"
        return "Sud/Isole"

    region_labels = ["Nord", "Centro", "Sud/Isole"]
    names = [k for k in (keys or results.keys()) if k in results and not results[k].empty]
    if not names:
        raise ValueError("No non-empty results to plot.")
    missing = [n for n in names if n not in labels]
    if missing:
        raise ValueError(f"`labels` is missing an explicit caption for: {missing}")

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), squeeze=False)

    for ax, name in zip(axes[0], names):
        df = compute_errors_km(results[name])
        true_r = df["lat_true"].map(macro_region)
        pred_r = df["lat_pred"].map(macro_region)
        cm = confusion_matrix(true_r, pred_r, labels=region_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=region_labels, yticklabels=region_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(labels[name])

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix: {output_path}")
    return fig


def plot_worst_cases_map(
    reference_df: pd.DataFrame,
    compare_df: Optional[pd.DataFrame],
    n_worst: int = 20,
    output_path: Optional[str] = None,
    reference_label: str = "Fine-tuned",
    compare_label: str = "Baseline",
) -> str:
    """Folium map with lines from GT to predictions for worst failures."""
    import folium

    work = compute_errors_km(reference_df)
    if work.empty:
        raise ValueError("reference_df has no valid predictions for worst-case map")

    work = work.nlargest(min(n_worst, len(work)), "error_km").rename(
        columns={"lat_pred": "lat_pred_ref", "lon_pred": "lon_pred_ref", "error_km": "error_km_ref"}
    )
    m = folium.Map(location=[42.5, 12.5], zoom_start=6, tiles="OpenStreetMap")

    if compare_df is not None and not compare_df.empty:
        cmp_work = compute_errors_km(compare_df)
        if not cmp_work.empty:
            cmp_slim = cmp_work[["image_id", "lat_pred", "lon_pred", "error_km"]].rename(
                columns={"lat_pred": "lat_pred_cmp", "lon_pred": "lon_pred_cmp", "error_km": "error_km_cmp"}
            )
            merged = work.merge(cmp_slim, on="image_id", how="left")
        else:
            merged = work.copy()
            merged["lat_pred_cmp"] = np.nan
            merged["lon_pred_cmp"] = np.nan
            merged["error_km_cmp"] = np.nan
    else:
        merged = work.copy()
        merged["lat_pred_cmp"] = np.nan
        merged["lon_pred_cmp"] = np.nan
        merged["error_km_cmp"] = np.nan

    for _, row in merged.iterrows():
        t_lat, t_lon = float(row["lat_true"]), float(row["lon_true"])
        r_lat, r_lon = float(row["lat_pred_ref"]), float(row["lon_pred_ref"])
        err_ref = float(row["error_km_ref"])
        folium.CircleMarker(
            [t_lat, t_lon], radius=4, color="green", fill=True, popup="Ground truth"
        ).add_to(m)
        folium.CircleMarker(
            [r_lat, r_lon], radius=4, color="red", fill=True,
            popup=f"{reference_label}: {err_ref:.0f} km",
        ).add_to(m)
        folium.PolyLine(
            [(t_lat, t_lon), (r_lat, r_lon)], color="red", weight=2, opacity=0.8
        ).add_to(m)
        if pd.notna(row.get("lat_pred_cmp")):
            c_lat, c_lon = float(row["lat_pred_cmp"]), float(row["lon_pred_cmp"])
            folium.CircleMarker(
                [c_lat, c_lon], radius=4, color="blue", fill=True,
                popup=f"{compare_label}: {row.get('error_km_cmp', float('nan')):.0f} km",
            ).add_to(m)
            folium.PolyLine(
                [(t_lat, t_lon), (c_lat, c_lon)], color="blue", weight=1.5, opacity=0.6, dash_array="6"
            ).add_to(m)

    legend_html = (
        "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;"
        "background:white;padding:10px;border:1px solid #888;font-size:12px'>"
        "<b>Worst-case legend</b><br>"
        "&#9679; Green = ground truth<br>"
        f"&#9679; Red = {reference_label}<br>"
        f"&#9679; Blue = {compare_label} (dashed line)"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    path = output_path or "worst_cases_map.html"
    m.save(path)
    print(f"Saved failure map: {path}")
    return path


def save_results_bundle(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    dataset_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for model_name, df in results.items():
        safe = model_name.replace("/", "_")
        csv_path = os.path.join(output_dir, f"{dataset_name}_{safe}_predictions.csv")
        out = compute_errors_km(df) if not df.empty else df
        out.to_csv(csv_path, index=False)
    summary_rows = []
    for model_name, df in results.items():
        m = compute_metrics(df)
        summary_rows.append({"dataset": dataset_name, "model": model_name, **m.to_dict()})
    summary_path = os.path.join(output_dir, f"{dataset_name}_metrics_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved metrics summary: {summary_path}")
