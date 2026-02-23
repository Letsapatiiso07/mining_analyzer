import argparse
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
IMAGERY = ROOT / "data" / "imagery"
MODELS  = ROOT / "models"
PREDS   = PROC / "predictions"
PREDS.mkdir(parents=True, exist_ok=True)
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

# ── Import model classes from train.py ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import UNet, ProductionLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Site definitions (kept local to avoid circular imports) ───────────────────
SA_SITES = {
    "Witwatersrand_Gold":  {"mineral": "Gold",     "region": "Gauteng"},
    "Rustenburg_Platinum": {"mineral": "Platinum",  "region": "North West"},
    "Sishen_Iron":         {"mineral": "Iron",      "region": "Northern Cape"},
    "Palabora_Copper":     {"mineral": "Copper",    "region": "Limpopo"},
    "Grootegeluk_Coal":    {"mineral": "Coal",      "region": "Limpopo"},
}

CLASS_COLORS = np.array([
    [128, 128, 128, 180],
    [220,  50,  50, 230],
    [255, 165,   0, 230],
    [ 60, 180,  60, 200],
], dtype=np.uint8)

NUM_CLASSES = 4


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation inference
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(site_id: str) -> dict:
    """Run U-Net inference on a site's raster. Returns metrics dict."""
    raster_path = IMAGERY / f"{site_id}_sentinel2.tif"
    if not raster_path.exists():
        log.error(f"Raster not found: {raster_path}")
        return {}

    model_path = MODELS / "unet_best.pt"
    if not model_path.exists():
        model_path = MODELS / "unet_final.pt"
    if not model_path.exists():
        log.warning("No trained U-Net found. Using label-based fallback prediction.")
        return _fallback_segmentation(site_id)

    model = UNet(in_channels=4, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    with rasterio.open(raster_path) as src:
        img       = src.read().astype(np.float32)
        transform = src.transform
        crs       = src.crs
        H, W      = src.height, src.width

    TILE      = 64
    pred_map  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                patch    = img[:, y:y + TILE, x:x + TILE]
                ph, pw   = patch.shape[1], patch.shape[2]
                if ph < TILE or pw < TILE:
                    patch = np.pad(patch, ((0, 0), (0, TILE - ph), (0, TILE - pw)))
                t      = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                logits = model(t)
                probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_map[:, y:y + ph, x:x + pw] += probs[:, :ph, :pw]
                count_map[y:y + ph, x:x + pw]   += 1

    pred_map /= np.maximum(count_map, 1)
    seg = pred_map.argmax(axis=0).astype(np.uint8)

    _save_segmentation_outputs(site_id, img, seg, transform, crs)
    return _compute_seg_metrics(site_id, seg)


def _fallback_segmentation(site_id: str) -> dict:
    """Use pre-generated label mask as fallback prediction."""
    lbl_path = IMAGERY / f"{site_id}_labels.tif"
    if not lbl_path.exists():
        return {}
    with rasterio.open(lbl_path) as src:
        seg       = src.read(1)
        transform = src.transform
        crs       = src.crs

    raster_path = IMAGERY / f"{site_id}_sentinel2.tif"
    with rasterio.open(raster_path) as src:
        img = src.read().astype(np.float32)

    _save_segmentation_outputs(site_id, img, seg, transform, crs)
    return _compute_seg_metrics(site_id, seg)


def _save_segmentation_outputs(site_id, img, seg, transform, crs):
    """Save segmentation mask GeoTIFF + RGB overlay PNG."""
    tif_out = PREDS / f"{site_id}_segmentation.tif"
    with rasterio.open(tif_out, "w", driver="GTiff",
                       height=seg.shape[0], width=seg.shape[1],
                       count=1, dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(seg, 1)

    H, W      = seg.shape
    rgb       = np.stack([img[0], img[1], img[2]], axis=-1)
    rgb_norm  = np.clip(rgb / (rgb.max() + 1e-6), 0, 1)

    overlay = CLASS_COLORS[seg, :3].astype(np.float32) / 255
    alpha   = CLASS_COLORS[seg, 3:4].astype(np.float32) / 255
    blended = np.clip(rgb_norm * (1 - alpha * 0.7) + overlay * alpha * 0.7, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb_norm);  axes[0].set_title("RGB Composite");        axes[0].axis("off")
    axes[1].imshow(blended);   axes[1].set_title("Segmentation Overlay"); axes[1].axis("off")

    labels  = ["Background", "Mine Pit", "Disturbed", "Vegetation"]
    colors  = [CLASS_COLORS[i, :3] / 255 for i in range(4)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(4)]
    axes[1].legend(handles, labels, loc="lower right", fontsize=8)

    fig.suptitle(f"{site_id.replace('_', ' ')} - Segmentation", fontsize=13, weight="bold")
    plt.tight_layout()
    png_out = PREDS / f"{site_id}_segmentation.png"
    plt.savefig(png_out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [OK] Segmentation outputs -> {tif_out.name}, {png_out.name}")


def _compute_seg_metrics(site_id: str, seg: np.ndarray) -> dict:
    total   = seg.size
    metrics = {"site_id": site_id}
    names   = ["background", "mine_pit", "disturbed", "vegetation"]
    for i, name in enumerate(names):
        metrics[f"{name}_frac"] = round(float((seg == i).sum() / total), 4)
    metrics["disturbed_ha"] = round((seg == 2).sum() * 0.01, 1)
    metrics["pit_ha"]       = round((seg == 1).sum() * 0.01, 1)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Production forecasting
# ─────────────────────────────────────────────────────────────────────────────

def run_production_forecast(site_id: str, horizon: int = 5) -> pd.DataFrame:
    """Forecast production for next `horizon` years using LSTM (or Prophet fallback)."""
    lstm_path    = MODELS / f"lstm_{site_id}.pt"
    prophet_path = MODELS / f"prophet_{site_id}.pkl"
    scaler_path  = MODELS / "production_scaler.pkl"

    prod_df = pd.read_csv(PROC / "production_timeseries.csv")
    hist    = prod_df[prod_df["site_id"] == site_id].sort_values("year")

    if hist.empty:
        log.warning(f"No production history for {site_id}")
        return pd.DataFrame()

    last_year    = int(hist["year"].max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    forecasts    = []

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if lstm_path.exists() and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model = ProductionLSTM()
        model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        model.to(DEVICE).eval()

        series        = hist["production"].values.reshape(-1, 1)
        series_scaled = scaler.transform(series).flatten().tolist()

        SEQ           = 5
        preds_scaled  = []
        buffer        = series_scaled[-SEQ:]

        with torch.no_grad():
            for _ in range(horizon):
                x    = torch.tensor(buffer[-SEQ:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                yhat = model(x).item()
                preds_scaled.append(yhat)
                buffer.append(yhat)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        for yr, val in zip(future_years, preds):
            forecasts.append({"year": yr, "forecast_lstm": round(float(max(0, val)), 1)})

    # ── Prophet ───────────────────────────────────────────────────────────────
    if prophet_path.exists():
        with open(prophet_path, "rb") as f:
            prophet_model = pickle.load(f)
        future  = prophet_model.make_future_dataframe(periods=horizon, freq="YS")
        fc      = prophet_model.predict(future)
        fc_sub  = fc[fc["ds"].dt.year.isin(future_years)].set_index("ds")
        for yr in future_years:
            row_match = fc_sub[fc_sub.index.year == yr]
            if not row_match.empty:
                val = float(row_match["yhat"].iloc[0])
                idx = next((i for i, f in enumerate(forecasts) if f["year"] == yr), None)
                if idx is not None:
                    forecasts[idx]["forecast_prophet"] = round(max(0, val), 1)
                else:
                    forecasts.append({"year": yr, "forecast_prophet": round(max(0, val), 1)})

    fc_df = pd.DataFrame(forecasts)
    if fc_df.empty:
        log.warning(f"No forecast models available for {site_id}")
        return pd.DataFrame()

    hist_out      = hist[["year", "production", "mineral", "unit"]].copy()
    hist_out["type"] = "historical"
    out_path      = PREDS / f"{site_id}_forecast.csv"
    fc_df.to_csv(out_path, index=False)

    _plot_forecast(site_id, hist_out, fc_df)
    log.info(f"  [OK] Forecast -> {out_path.name}")
    return fc_df


def _plot_forecast(site_id: str, hist: pd.DataFrame, fc: pd.DataFrame):
    fig, ax  = plt.subplots(figsize=(10, 5))
    unit     = hist["unit"].iloc[0]    if "unit"    in hist.columns else ""
    mineral  = hist["mineral"].iloc[0] if "mineral" in hist.columns else ""

    ax.plot(hist["year"], hist["production"], "o-", color="#2196F3",
            linewidth=2, markersize=5, label="Historical")

    if "forecast_lstm" in fc.columns:
        ax.plot(fc["year"], fc["forecast_lstm"], "s--", color="#FF5722",
                linewidth=2, markersize=6, label="LSTM Forecast")

    if "forecast_prophet" in fc.columns:
        ax.plot(fc["year"], fc["forecast_prophet"], "^--", color="#4CAF50",
                linewidth=2, markersize=6, label="Prophet Forecast")

    ax.axvline(hist["year"].max(), color="grey", linestyle=":", alpha=0.7)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Production ({unit})", fontsize=11)
    ax.set_title(f"{site_id.replace('_', ' ')} - {mineral} Production Forecast",
                 fontsize=13, weight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PREDS / f"{site_id}_forecast.png", dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Environmental analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_environmental_analysis(site_id: str) -> dict:
    ndvi_ts  = pd.read_csv(PROC / "ndvi_timeseries.csv")
    site_ts  = ndvi_ts[ndvi_ts["site_id"] == site_id].sort_values("year")

    if site_ts.empty:
        return {}

    c             = np.polyfit(site_ts["year"], site_ts["ndvi_mean"], 1)
    trend_per_yr  = c[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(site_ts["year"], site_ts["ndvi_p10"], site_ts["ndvi_p90"],
                    alpha=0.2, color="#4CAF50", label="P10-P90")
    ax.plot(site_ts["year"], site_ts["ndvi_mean"], "o-", color="#2E7D32",
            linewidth=2, label="Mean NDVI")
    ax.plot(site_ts["year"], np.polyval(c, site_ts["year"]), "--", color="red",
            alpha=0.7, label=f"Trend {trend_per_yr:+.4f}/yr")
    ax.set_ylim(-0.1, 0.6)
    ax.set_xlabel("Year"); ax.set_ylabel("NDVI")
    ax.set_title(f"{site_id.replace('_', ' ')} - NDVI Trend (vegetation health)", weight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PREDS / f"{site_id}_ndvi_trend.png", dpi=120)
    plt.close()
    log.info("  [OK] NDVI trend plot saved")

    return {
        "site_id":             site_id,
        "ndvi_trend_per_yr":   round(trend_per_yr, 5),
        "ndvi_2024":           round(float(site_ts.iloc[-1]["ndvi_mean"]), 4),
        "veg_loss_2015_2024":  round(
            float(site_ts.iloc[0]["veg_frac"] - site_ts.iloc[-1]["veg_frac"]) * 100, 2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def predict_site(site_id: str):
    log.info(f"-- Analyzing site: {site_id} --")
    seg_metrics = run_segmentation(site_id)
    fc          = run_production_forecast(site_id)
    env         = run_environmental_analysis(site_id)
    summary     = {**seg_metrics, **env}
    log.info(f"  Summary: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run inference on mine sites")
    parser.add_argument("--site", default="all",
                        help="Site ID or 'all'. E.g. Rustenburg_Platinum")
    args = parser.parse_args()

    sites = list(SA_SITES.keys()) if args.site == "all" else [args.site]

    log.info("=" * 60)
    log.info("Step 4: Inference")
    log.info("=" * 60)

    all_summaries = []
    for s in sites:
        try:
            summary = predict_site(s)
            all_summaries.append(summary)
        except Exception as e:
            log.error(f"  Error on {s}: {e}")

    if all_summaries:
        out = PREDS / "all_sites_summary.csv"
        pd.DataFrame(all_summaries).to_csv(out, index=False)
        log.info(f"[OK] Summary -> {out}")

    log.info("=" * 60)
    log.info("[DONE] Inference complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()