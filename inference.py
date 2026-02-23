import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent
PROC    = ROOT / "data" / "processed"
IMAGERY = ROOT / "data" / "imagery"
MODELS  = ROOT / "models"
PREDS   = PROC / "predictions"
PREDS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# ── Label colours: background, pit, disturbed, vegetation ────────────────────
CLASS_COLORS = np.array([
    [128, 128, 128],   # background – grey
    [220,  50,  50],   # mine pit   – red
    [255, 165,   0],   # disturbed  – orange
    [ 60, 180,  60],   # vegetation – green
], dtype=np.uint8)

CLASS_NAMES = ["background", "mine_pit", "disturbed", "vegetation"]


# ── Model definitions ────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_features=32):
        super().__init__()
        f = base_features
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _DoubleConv(f, f*2)
        self.enc3 = _DoubleConv(f*2, f*4)
        self.enc4 = _DoubleConv(f*4, f*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(f*8, f*16)
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.dec4 = _DoubleConv(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.dec3 = _DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.dec2 = _DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1 = _DoubleConv(f*2, f)
        self.out_conv = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.out_conv(d1)


class ProductionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment_site(site_id: str, image_array: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns: (rgb_image, segmentation_mask, metrics_dict)
    image_array: optional (4,H,W) float32 override (e.g. user-uploaded image)
    """
    # Load or use provided image
    if image_array is not None:
        img = image_array.astype(np.float32)
    else:
        raster_path = IMAGERY / f"{site_id}_sentinel2.tif"
        if not raster_path.exists():
            raise FileNotFoundError(f"No raster found for site {site_id}")
        with rasterio.open(raster_path) as src:
            img = src.read().astype(np.float32)   # (4,H,W)

    H, W = img.shape[1], img.shape[2]

    # Try trained model, else use label file as proxy
    model_path = MODELS / "unet_best.pt"
    if not model_path.exists():
        model_path = MODELS / "unet_final.pt"

    if model_path.exists():
        seg = _unet_predict(img, model_path, H, W)
    else:
        lbl_path = IMAGERY / f"{site_id}_labels.tif"
        if lbl_path.exists():
            with rasterio.open(lbl_path) as src:
                seg = src.read(1)
        else:
            # Fallback: simple NDVI threshold segmentation
            seg = _ndvi_threshold_seg(img)

    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
    rgb = np.clip(rgb / (rgb.max() + 1e-6), 0, 1)
    metrics = _seg_metrics(seg)
    return rgb, seg, metrics


def _unet_predict(img: np.ndarray, model_path: Path, H: int, W: int) -> np.ndarray:
    model = UNet(in_channels=4, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    TILE = 64
    pred_map  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                patch = img[:, y:y+TILE, x:x+TILE]
                ph, pw = patch.shape[1], patch.shape[2]
                padded = np.pad(patch, ((0,0),(0,TILE-ph),(0,TILE-pw)))
                t = torch.from_numpy(padded).unsqueeze(0).to(DEVICE)
                probs = F.softmax(model(t), dim=1).squeeze(0).cpu().numpy()
                pred_map[:, y:y+ph, x:x+pw] += probs[:, :ph, :pw]
                count_map[y:y+ph, x:x+pw] += 1
    pred_map /= np.maximum(count_map, 1)
    return pred_map.argmax(0).astype(np.uint8)


def _ndvi_threshold_seg(img: np.ndarray) -> np.ndarray:
    """Rough NDVI-based segmentation as last resort fallback."""
    nir, red = img[3], img[0]
    ndvi = (nir - red) / (nir + red + 1e-8)
    seg = np.zeros(ndvi.shape, dtype=np.uint8)
    seg[ndvi >= 0.25] = 3   # vegetation
    seg[(ndvi >= 0.05) & (ndvi < 0.25)] = 2  # disturbed
    seg[ndvi < 0.05] = 1    # mine pit
    return seg


def _seg_metrics(seg: np.ndarray) -> dict:
    total = seg.size
    m = {}
    for i, name in enumerate(CLASS_NAMES):
        frac = (seg == i).sum() / total
        m[f"{name}_pct"] = round(float(frac) * 100, 1)
    m["disturbed_ha"] = round((seg == 2).sum() * 0.01, 1)
    m["pit_ha"]       = round((seg == 1).sum() * 0.01, 1)
    return m


def make_overlay(rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend RGB and segmentation colour map into a single RGBA image (0-255 uint8)."""
    overlay = CLASS_COLORS[seg].astype(np.float32) / 255   # (H,W,3)
    rgb_f   = rgb.astype(np.float32)
    blended = rgb_f * (1 - alpha) + overlay * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return blended


# ── Production forecasting ────────────────────────────────────────────────────

def forecast_production(site_id: str, horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns: (historical_df, forecast_df)
    """
    prod_path   = PROC / "production_timeseries.csv"
    scaler_path = MODELS / "production_scaler.pkl"
    lstm_path   = MODELS / f"lstm_{site_id}.pt"
    prophet_path = MODELS / f"prophet_{site_id}.pkl"

    if not prod_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    hist = pd.read_csv(prod_path)
    hist = hist[hist["site_id"] == site_id].sort_values("year").reset_index(drop=True)

    if hist.empty:
        return hist, pd.DataFrame()

    last_year    = int(hist["year"].max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    forecasts    = {yr: {} for yr in future_years}

    # LSTM
    if lstm_path.exists() and scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            model = ProductionLSTM()
            model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
            model.to(DEVICE).eval()
            series_scaled = scaler.transform(
                hist["production"].values.reshape(-1, 1)
            ).flatten().tolist()
            SEQ = 5
            buf = series_scaled[-SEQ:]
            with torch.no_grad():
                for yr in future_years:
                    x = torch.tensor(buf[-SEQ:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                    yhat = model(x).item()
                    buf.append(yhat)
                    val = float(scaler.inverse_transform([[yhat]])[0, 0])
                    forecasts[yr]["LSTM"] = max(0, round(val, 1))
        except Exception as e:
            log.warning(f"LSTM forecast failed: {e}")

    # Prophet
    if prophet_path.exists():
        try:
            with open(prophet_path, "rb") as f:
                pm = pickle.load(f)
            future = pm.make_future_dataframe(periods=horizon, freq="YS")
            fc = pm.predict(future)
            for yr in future_years:
                row = fc[fc["ds"].dt.year == yr]
                if not row.empty:
                    forecasts[yr]["Prophet"] = max(0, round(float(row["yhat"].iloc[0]), 1))
        except Exception as e:
            log.warning(f"Prophet forecast failed: {e}")

    fc_df = pd.DataFrame([
        {"year": yr, **vals} for yr, vals in forecasts.items()
    ])

    return hist, fc_df


# ── NDVI trend ────────────────────────────────────────────────────────────────

def get_ndvi_timeseries(site_id: str) -> pd.DataFrame:
    path = PROC / "ndvi_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["site_id"] == site_id].sort_values("year").reset_index(drop=True)


def get_pit_metrics(site_id: str) -> pd.DataFrame:
    path = PROC / "pit_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["site_id"] == site_id].reset_index(drop=True)