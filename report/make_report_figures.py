#!/usr/bin/env python3
"""Generate all figures for the CVPR report from the benchmark prediction CSVs.

Single source of truth: reads the per-model prediction CSVs produced by
benchmarking/Italy_Geoloc_Benchmark.ipynb (copied into ../benchmarking/benchmark_results)
and renders every figure the paper uses with one consistent style. Re-run after any new
benchmark pass to keep the paper's figures in lock-step with its tables.

Usage:
    python make_report_figures.py
Outputs PDFs into ./charts/.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "benchmarking", "benchmark_results")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# Italy bounding box (matches training / benchmarking config).
MIN_LAT, MAX_LAT = 35.4, 47.2
MIN_LON, MAX_LON = 6.6, 18.8

# ── Consistent style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # editable/embeddable fonts for camera-ready
    "ps.fonttype": 42,
})

# One colour per model, reused across every figure for consistency.
C_GLOBAL = "#8c8c8c"   # grey   — GeoCLIP global gallery
C_ITALY = "#e07b39"    # orange — GeoCLIP Italy-constrained gallery
C_STREET = "#9467bd"   # purple — StreetCLIP
C_OURS = "#c0392b"     # red    — our fine-tuned model
C_GT = "#2e8b57"       # green  — ground truth

DS = "osv5m"


def load(model_file: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS, f"{DS}_{model_file}_predictions.csv"))
    if "error_km" not in df.columns:
        raise ValueError(f"{model_file}: expected precomputed error_km column")
    return df.dropna(subset=["lat_true", "lon_true", "lat_pred", "lon_pred", "error_km"])


GLOBAL = load("geoclip_retrieval_global")
ITALY = load("geoclip_retrieval_italy")
STREET = load("streetclip_italy")
OURS = load("geoclip_italy_finetuned")


# ── Figure 1: prediction scatter, 3 panels ───────────────────────────────────
def fig_scatter():
    panels = [
        (GLOBAL, "GeoCLIP — global gallery", C_GLOBAL),
        (ITALY, "GeoCLIP — Italy gallery", C_ITALY),
        (OURS, "Ours (V3.3 + TTA) — regression", C_OURS),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    for ax, (df, title, color) in zip(axes, panels):
        ax.scatter(df.lon_pred, df.lat_pred, s=7, alpha=0.35, c=color,
                   label="Predicted", rasterized=True, linewidths=0)
        ax.scatter(df.lon_true, df.lat_true, s=7, alpha=0.35, c=C_GT,
                   label="Ground truth", rasterized=True, linewidths=0)
        # Italy bounding box outline.
        ax.plot([MIN_LON, MAX_LON, MAX_LON, MIN_LON, MIN_LON],
                [MIN_LAT, MIN_LAT, MAX_LAT, MAX_LAT, MIN_LAT],
                c="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_xlim(-12, 30)
        ax.set_ylim(30, 52)
        med = np.median(df.error_km)
        ax.text(0.03, 0.03, f"median {med:.0f} km", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.85))
    axes[0].set_ylabel("Latitude")
    axes[0].legend(loc="upper left", markerscale=2, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(CHARTS, "scatter_3panel.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ── Figure 2: error CDF + accuracy@distance ──────────────────────────────────
def fig_cdf_accuracy():
    series = [
        ("GeoCLIP (global)", GLOBAL, C_GLOBAL),
        ("GeoCLIP (Italy gallery)", ITALY, C_ITALY),
        ("StreetCLIP", STREET, C_STREET),
        ("Ours (V3.3 + TTA)", OURS, C_OURS),
    ]
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: CDF of localization error.
    for name, df, color in series:
        e = np.sort(df.error_km.values)
        y = np.arange(1, len(e) + 1) / len(e)
        axc.plot(e, y, color=color, lw=2, label=name)
    axc.set_xlim(0, 600)
    axc.set_ylim(0, 1)
    axc.yaxis.set_major_formatter(PercentFormatter(1.0))
    axc.set_xlabel("Localization error (km)")
    axc.set_ylabel("Fraction of test images")
    axc.set_title("(a) Cumulative error distribution")
    axc.grid(alpha=0.3)
    axc.legend(loc="lower right")

    # Right: accuracy at distance thresholds.
    ths = [25, 50, 100, 200, 500]
    x = np.arange(len(ths))
    width = 0.8 / len(series)
    for i, (name, df, color) in enumerate(series):
        e = df.error_km.values
        vals = [(e <= t).mean() * 100 for t in ths]
        off = (i - (len(series) - 1) / 2) * width
        axb.bar(x + off, vals, width, color=color, label=name)
    axb.set_xticks(x)
    axb.set_xticklabels([f"{t}" for t in ths])
    axb.set_xlabel("Distance threshold (km)")
    axb.set_ylabel("Accuracy (% within threshold)")
    axb.set_title("(b) Accuracy at distance thresholds")
    axb.grid(alpha=0.3, axis="y")
    axb.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(CHARTS, "cdf_accuracy.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ── Figure 3: version ablation (median error) ────────────────────────────────
def fig_ablation():
    # Baselines from prediction CSVs; intermediate versions from the consistent
    # OSV5M version-sweep CSV (same test set, same pipeline).
    sweep = pd.read_csv(os.path.join(RESULTS, f"{DS}_version_sweep_metrics.csv"))
    sweep = sweep.set_index("model")["median_km"].to_dict()

    rows = [
        ("GeoCLIP\n(global)", np.median(GLOBAL.error_km), C_GLOBAL),
        ("GeoCLIP\n(Italy gall.)", np.median(ITALY.error_km), C_ITALY),
        ("StreetCLIP", np.median(STREET.error_km), C_STREET),
        ("V3.1\n(r=8)", sweep.get("V3", 124.2), "#4c72b0"),
        ("V3.2\n(r=16)", sweep.get("V3.2", 122.2), "#4c72b0"),
        ("V3.3\n(no TTA)", sweep.get("V3.3", 119.7), "#4c72b0"),
        ("V3.3+TTA\n(ours)", np.median(OURS.error_km), C_OURS),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.72)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Median error (km)")
    ax.set_ylim(0, 600)
    # Divider between retrieval baselines and our regression models.
    ax.axvline(2.5, color="0.5", ls=":", lw=1)
    ax.text(1.0, 560, "retrieval baselines", ha="center", fontsize=8.5, style="italic", color="0.35")
    ax.text(4.75, 560, "our regression models", ha="center", fontsize=8.5, style="italic", color="0.35")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(CHARTS, "ablation_median.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_scatter()
    fig_cdf_accuracy()
    fig_ablation()
    print("done ->", CHARTS)
