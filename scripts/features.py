import sys
import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
IMAGERY = ROOT / "data" / "imagery"
MODELS  = ROOT / "models"
MODELS.mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

# ── Logging (UTF-8) ───────────────────────────────────────────────────────────
_stream = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)
_stream.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_file = logging.FileHandler(ROOT / "logs" / "pipeline.log", encoding="utf-8")
_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_stream, _file])
log = logging.getLogger(__name__)

CLASS_NAMES = {0: "background", 1: "mine_pit", 2: "disturbed", 3: "vegetation"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pit extent metrics from label masks
# ─────────────────────────────────────────────────────────────────────────────

def extract_pit_metrics() -> pd.DataFrame:
    out = PROC / "pit_metrics.csv"
    if out.exists():
        log.info("[OK] pit_metrics.csv already exists.")
        return pd.read_csv(out)

    rows = []
    for mask_path in sorted(IMAGERY.glob("*_labels.tif")):
        site = mask_path.stem.replace("_labels", "")
        with rasterio.open(mask_path) as src:
            labels       = src.read(1)
            pixel_area_m2 = abs(src.res[0] * src.res[1]) * (111_320 ** 2)

        total_px = labels.size
        for cls_id, cls_name in CLASS_NAMES.items():
            count = (labels == cls_id).sum()
            rows.append({
                "site_id":  site,
                "class":    cls_name,
                "pixels":   int(count),
                "fraction": round(count / total_px, 4),
                "area_ha":  round(count * pixel_area_m2 / 10_000, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info(f"[OK] Pit metrics -> {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulated multi-year NDVI change
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ndvi_change_series() -> pd.DataFrame:
    """Simulate annual NDVI statistics per site over 2015-2024."""
    out = PROC / "ndvi_timeseries.csv"
    if out.exists():
        log.info("[OK] ndvi_timeseries.csv already exists.")
        return pd.read_csv(out)

    np.random.seed(7)
    rows = []

    for ndvi_path in sorted(IMAGERY.glob("*_ndvi.tif")):
        site = ndvi_path.stem.replace("_ndvi", "")
        with rasterio.open(ndvi_path) as src:
            base_ndvi = src.read(1)

        for yr in range(2015, 2025):
            dt         = yr - 2015
            degradation = -0.008 * dt
            annual     = base_ndvi + degradation + np.random.normal(0, 0.005, base_ndvi.shape)

            rows.append({
                "site_id":   site,
                "year":      yr,
                "ndvi_mean": float(np.mean(annual)),
                "ndvi_std":  float(np.std(annual)),
                "ndvi_p10":  float(np.percentile(annual, 10)),
                "ndvi_p90":  float(np.percentile(annual, 90)),
                "veg_frac":  float((annual > 0.25).mean()),
                "bare_frac": float((annual < 0.05).mean()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info(f"[OK] NDVI time-series -> {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Production feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_production_features() -> dict:
    """Create lagged / rolling features for each site's production time-series."""
    prod_path = PROC / "production_timeseries.csv"
    if not prod_path.exists():
        log.error("production_timeseries.csv not found - run download_and_process.py first")
        return {}

    df            = pd.read_csv(prod_path)
    site_datasets = {}

    for site, grp in df.groupby("site_id"):
        grp = grp.sort_values("year").copy()

        for lag in [1, 2, 3]:
            grp[f"prod_lag{lag}"] = grp["production"].shift(lag)

        grp["prod_roll3_mean"] = grp["production"].rolling(3, min_periods=1).mean()
        grp["prod_roll3_std"]  = grp["production"].rolling(3, min_periods=1).std().fillna(0)
        grp["prod_roll5_mean"] = grp["production"].rolling(5, min_periods=1).mean()
        grp["prod_yoy_pct"]    = grp["production"].pct_change().fillna(0)
        grp["year_idx"]        = grp["year"] - grp["year"].min()

        grp = grp.dropna().reset_index(drop=True)
        site_datasets[site] = grp

    combined = pd.concat(site_datasets.values(), ignore_index=True)
    combined.to_csv(PROC / "production_features.csv", index=False)
    log.info(f"[OK] Production features -> {PROC / 'production_features.csv'}")

    scaler = MinMaxScaler()
    scaler.fit(combined[["production"]])
    with open(MODELS / "production_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("[OK] Scaler saved -> models/production_scaler.pkl")

    return site_datasets


# ─────────────────────────────────────────────────────────────────────────────
# 4. Segmentation dataset index
# ─────────────────────────────────────────────────────────────────────────────

def build_segmentation_dataset_index() -> pd.DataFrame:
    """Index all image / label pairs for the U-Net training loop."""
    out = PROC / "segmentation_index.csv"
    if out.exists():
        log.info("[OK] segmentation_index.csv already exists.")
        return pd.read_csv(out)

    rows = []
    for img_path in sorted(IMAGERY.glob("*_sentinel2.tif")):
        site = img_path.stem.replace("_sentinel2", "")
        lbl  = IMAGERY / f"{site}_labels.tif"
        ndvi = IMAGERY / f"{site}_ndvi.tif"
        rows.append({
            "site_id":    site,
            "image_path": str(img_path),
            "label_path": str(lbl)  if lbl.exists()  else "",
            "ndvi_path":  str(ndvi) if ndvi.exists() else "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info(f"[OK] Segmentation index ({len(df)} samples) -> {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Step 2: Feature engineering")
    log.info("=" * 60)

    metrics = extract_pit_metrics()
    log.info(f"  Pit metrics shape: {metrics.shape}")

    ndvi_ts = simulate_ndvi_change_series()
    log.info(f"  NDVI time-series shape: {ndvi_ts.shape}")

    prod_feats = engineer_production_features()
    log.info(f"  Production feature datasets: {list(prod_feats.keys())}")

    seg_idx = build_segmentation_dataset_index()
    log.info(f"  Segmentation samples: {len(seg_idx)}")

    log.info("=" * 60)
    log.info("[DONE] Feature engineering complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()