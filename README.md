# SA Open-Pit Mining Site Analyzer

An AI-powered, fully local Python application for analyzing South African open-pit mining sites using satellite imagery and historical data. No cloud services, no Docker — runs entirely on your machine.

---

## Project Overview

| Feature | Details |
|---|---|
| **Target Region** | South Africa (Witwatersrand Gold, Rustenburg Platinum, Sishen Iron, Palabora Copper, Grootegeluk Coal) |
| **Computer Vision** | U-Net semantic segmentation (4-class: background / pit / disturbed / vegetation) |
| **Forecasting** | LSTM + Prophet production trend prediction (1990–2024 + 5-year horizon) |
| **Remote Sensing** | Sentinel-2 proxy rasters, NDVI, EVI, Bare Soil Index |
| **Dashboard** | Streamlit with Folium maps, Plotly charts, custom image upload |
| **Experiment Tracking** | MLflow (local `./mlruns` store) |

---

## Project Structure

```
mining_analyzer/
├── app.py                         # Streamlit dashboard
├── inference.py                   # Shared inference utilities
├── run_pipeline.py                # Convenience full-pipeline runner
├── requirements.txt
├── setup.sh                       # One-command setup (Linux/macOS)
├── README.md
│
├── scripts/
│   ├── 01_download_and_process.py # Data ingestion & synthetic raster generation
│   ├── 02_features.py             # Feature engineering (NDVI, pit metrics, lags)
│   ├── 03_train.py                # U-Net + LSTM + Prophet training
│   ├── 04_predict.py              # Inference: segmentation + forecasting
│   └── train_helpers.py           # Shared model class definitions
│
├── data/
│   ├── raw/                       # Raw downloaded files
│   ├── processed/                 # Cleaned CSVs, GeoJSON, features
│   │   └── predictions/           # Inference outputs
│   └── imagery/                   # GeoTIFF rasters + label masks
│
├── models/                        # Saved PyTorch + Prophet models
│   ├── unet_best.pt
│   ├── lstm_<site>.pt
│   ├── prophet_<site>.pkl
│   └── production_scaler.pkl
│
├── mlruns/                        # MLflow experiment logs (auto-created)
└── logs/                          # Pipeline logs (auto-created)
```

---

## Quick Start

### Option A — Automated Setup (Linux/macOS)

```bash
git clone <repo> mining_analyzer
cd mining_analyzer
bash setup.sh
# After setup completes:
source venv/bin/activate
streamlit run app.py
```

### Option B — Manual Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline (each script is idempotent — safe to re-run)
python scripts/01_download_and_process.py   # ~30 sec
python scripts/02_features.py               # ~10 sec
python scripts/03_train.py                  # ~2-5 min (CPU) / ~30 sec (GPU)
python scripts/04_predict.py               # ~1 min

# 4. Launch dashboard
streamlit run app.py
```

### Option C — Single command runner

```bash
python run_pipeline.py            # runs all 4 steps
python run_pipeline.py --step 3  # run training only
```

---

## Model Details

### U-Net Segmentation
- **Architecture**: Encoder-decoder with skip connections, 4-level depth
- **Input**: 4-band Sentinel-2 tile (R, G, B, NIR), 64×64 patches
- **Output**: 4-class pixel mask (background / mine pit / disturbed zone / vegetation)
- **Loss**: CrossEntropy + Dice (0.5 weight each)
- **Optimizer**: Adam with Cosine Annealing LR
- **Training**: 8 epochs, ~5 sites × 32 patches/image
- **GPU/CPU**: Automatic fallback to CPU if no CUDA available

### LSTM Forecaster (per site)
- **Architecture**: 2-layer LSTM, hidden=64, dropout=0.2
- **Input**: Sliding window of 5 years' scaled production values
- **Output**: Next year's production
- **Training**: 30 epochs, MSE loss, Adam optimizer
- **Inference**: Autoregressive multi-step forecasting

### Prophet Forecaster (per site)
- **Model**: Facebook Prophet with multiplicative seasonality
- **Fallback**: Used alongside LSTM for comparison in dashboard

---

## Dashboard Features

| Tab | Content |
|---|---|
| Site Map | Interactive Folium map with all SA mine sites + colour-coded minerals |
| Imagery | RGB composite, NDVI map, pre-computed segmentation side-by-side |
| Segmentation | Blended overlay + land cover pie chart + area metrics |
| Production | Historical trend + LSTM/Prophet forecasts (configurable horizon) |
| Environment | NDVI degradation trend, vegetation/bare soil charts, risk matrix |
| Upload | Drag-and-drop custom GeoTIFF or PNG/JPG for live segmentation |

---

## Data Sources

| Dataset | Source | Notes |
|---|---|---|
| Mine site locations | Derived from public USGS MRDS | 5 SA sites hardcoded |
| Production history | Synthetic (statistically realistic 1990–2024) | Mirrors real mine trajectories |
| Sentinel-2 imagery | Synthetic proxy GeoTIFFs | Real tiles require free ESA Copernicus account |
| Label masks | Auto-generated from pit geometry | Suitable for model training demo |
| NDVI time-series | Computed from synthetic rasters | Progressive degradation model |

### Getting Real Sentinel-2 Tiles
1. Register at [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
2. Use `sentinelsat` Python package:
   ```bash
   pip install sentinelsat
   sentinelsat --user USER --password PASS \
     --geometry POLYGON \
     --start 20230101 --end 20231231 \
     --producttype S2MSI2A --download
   ```
3. Replace synthetic rasters in `data/imagery/` with real tiles (same naming convention)

---

## Configuration

Edit `scripts/01_download_and_process.py` to add new mine sites:

```python
SA_SITES["My_New_Mine"] = {
    "lon": 25.0, "lat": -28.5,
    "description": "My mine description",
    "mineral": "Gold",
    "region": "Free State",
    "area_km2": 75,
    "production_start": 1995,
}
```

---

## MLflow Experiment Tracking

View training metrics locally:
```bash
mlflow ui --backend-store-uri ./mlruns
# Opens at http://localhost:5000
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Storage | 2 GB free | 5 GB free |
| GPU | Optional (CPU works) | NVIDIA CUDA GPU |
| Python | 3.10+ | 3.11 |

---

## Contributing

1. Fork and clone
2. Create feature branch: `git checkout -b feature/my-feature`
3. Run pipeline to verify: `python run_pipeline.py`
4. Submit PR

---

## License

MIT License. Synthetic data and models are for research/demonstration purposes.