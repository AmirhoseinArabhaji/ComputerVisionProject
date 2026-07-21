# GeoCLIP-Italy

Fine-tuning [GeoCLIP](https://github.com/VicenteVivan/geo-clip) to predict GPS coordinates from photos taken in Italy, using LoRA adapters, a custom regression head, and a land-aware geographic penalty loss.

Given a single photo, the model predicts a `(latitude, longitude)` pair anywhere within Italy's bounding box — no reference gallery or retrieval step required.

## Results

Evaluated on a fixed OSV5M Italy test set (1,397 held-out images):

| Model | Mean error | Median error | Within 200 km | Within 500 km |
|---|---|---|---|---|
| Baseline GeoCLIP (retrieval) | 1290.5 km | 538.3 km | 24.6% | 47.2% |
| **GeoCLIP-Italy (fine-tuned)** | **195.2 km** | **124.2 km** | **66.1%** | **91.6%** |

Full benchmark methodology, ablations, and figures are in [`report/report.pdf`](report/report.pdf) and [`benchmarking/`](benchmarking/).

## Approach

- **Backbone**: OpenAI CLIP ViT-Large-Patch14, via GeoCLIP's pretrained image encoder — frozen.
- **Adaptation**: LoRA (rank=8, alpha=16) injected into `q_proj`/`v_proj` of every attention layer, plus a fully unfrozen `visual_projection` + MLP for domain adaptation. ~3.74M trainable params out of 440M (0.85%).
- **Regression head**: `512 → 1024 → 512 → 256 → 2` (BatchNorm, GELU, dropout, sigmoid output), predicting normalized coordinates mapped back to Italy's lat/lon bounding box.
- **Loss**: a weighted combination of L1 and haversine distance, plus a differentiable land-penalty term (built from a coastline distance-transform grid) that discourages predicting GPS points out at sea.
- **Data**: [OSV5M](https://github.com/gastruc/osv5m), filtered to Italy (~112K train/val images, 1.4K fixed test images).

See the training notebook for exact hyperparameters, dataset splits, and the full model definition.

## Repository layout

```
GeoClip_Training_V3_3.ipynb      # fine-tuning pipeline (Colab, GPU required)
GeoClip_Evaluation_V3_3.ipynb    # evaluation against the fixed OSV5M test set
osv5m/                           # dataset preprocessing (filtering, bucketing, packaging)
scraper/                         # legacy GPS/image scraping pipeline (Google Landmarks-based)
benchmarking/                    # standalone benchmarking suite (metrics, plots, model comparisons)
report/                          # write-up (report.tex / report.pdf) and result figures
```

## Running it

All notebooks are designed to run top-to-bottom on **Google Colab** (GPU) with **Google Drive** for persistent storage — each notebook installs its own dependencies in the first cell. A local Python environment can be used for reading/editing code, but training and evaluation require a GPU.

Core dependencies: `geoclip`, `torch`, `torchvision`, `peft`, `pandas`, `scikit-learn`, `folium`, `global-land-mask`, `scipy`.

## Acknowledgments

Built on top of [GeoCLIP](https://github.com/VicenteVivan/geo-clip) and trained on [OSV5M](https://github.com/gastruc/osv5m). Legacy experiments used [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark).
