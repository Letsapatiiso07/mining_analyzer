import sys
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score

warnings.filterwarnings("ignore")

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

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
log.info(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# A. U-Net Segmentation
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    """Lightweight U-Net for 4-band satellite imagery -> 4-class segmentation."""

    def __init__(self, in_channels=4, num_classes=4, base_features=32):
        super().__init__()
        f = base_features

        self.enc1      = DoubleConv(in_channels, f)
        self.enc2      = DoubleConv(f,     f * 2)
        self.enc3      = DoubleConv(f * 2, f * 4)
        self.enc4      = DoubleConv(f * 4, f * 8)
        self.pool      = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up4  = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3  = nn.ConvTranspose2d(f * 8,  f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8,  f * 4)
        self.up2  = nn.ConvTranspose2d(f * 4,  f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4,  f * 2)
        self.up1  = nn.ConvTranspose2d(f * 2,  f,     2, stride=2)
        self.dec1 = DoubleConv(f * 2,  f)

        self.out_conv = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class MineDataset(Dataset):
    """Loads 4-band GeoTIFF + label mask pairs with random patch extraction."""

    PATCH = 64

    def __init__(self, index_df: pd.DataFrame, augment: bool = False, patches_per_image: int = 16):
        self.records          = index_df[index_df["label_path"] != ""].reset_index(drop=True)
        self.augment          = augment
        self.patches_per_image = patches_per_image
        self._build_patch_list()

    def _build_patch_list(self):
        self.patches = []
        np.random.seed(42)
        for _, row in self.records.iterrows():
            for _ in range(self.patches_per_image):
                y = np.random.randint(0, 256 - self.PATCH)
                x = np.random.randint(0, 256 - self.PATCH)
                self.patches.append((row["image_path"], row["label_path"], y, x))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, lbl_path, y, x = self.patches[idx]
        P = self.PATCH

        with rasterio.open(img_path) as src:
            img = src.read(window=rasterio.windows.Window(x, y, P, P)).astype(np.float32)

        with rasterio.open(lbl_path) as src:
            lbl = src.read(1, window=rasterio.windows.Window(x, y, P, P)).astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()

        return torch.from_numpy(img), torch.from_numpy(lbl)


def dice_loss(pred, target, num_classes=4, smooth=1.0):
    pred_soft  = torch.softmax(pred, dim=1)
    total_loss = 0.0
    for cls in range(num_classes):
        p            = pred_soft[:, cls]
        t            = (target == cls).float()
        intersection = (p * t).sum()
        total_loss  += 1 - (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return total_loss / num_classes


def train_unet(epochs: int = 8, batch_size: int = 8, lr: float = 3e-4):
    log.info("-- Training U-Net --")
    idx = pd.read_csv(PROC / "segmentation_index.csv")

    if len(idx) == 0:
        log.error("No segmentation data found. Run scripts 01 and 02 first.")
        return

    train_ds = MineDataset(idx, augment=True,  patches_per_image=32)
    val_ds   = MineDataset(idx, augment=False, patches_per_image=8)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = UNet(in_channels=4, num_classes=NUM_CLASSES).to(DEVICE)
    ce_loss   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("UNet_Segmentation")

    iou = 0.0  # ensure defined even if epochs=0
    with mlflow.start_run(run_name="unet_train"):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr,
                           "num_classes": NUM_CLASSES, "device": str(DEVICE)})

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for imgs, lbls in train_dl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                preds = model(imgs)
                loss  = ce_loss(preds, lbls) + 0.5 * dice_loss(preds, lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses, all_preds, all_lbls = [], [], []
            with torch.no_grad():
                for imgs, lbls in val_dl:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    preds = model(imgs)
                    loss  = ce_loss(preds, lbls) + 0.5 * dice_loss(preds, lbls)
                    val_losses.append(loss.item())
                    all_preds.extend(preds.argmax(1).cpu().numpy().flatten().tolist())
                    all_lbls.extend(lbls.cpu().numpy().flatten().tolist())

            train_l = np.mean(train_losses)
            val_l   = np.mean(val_losses)
            iou     = jaccard_score(all_lbls, all_preds, average="macro", zero_division=0)

            mlflow.log_metrics({"train_loss": train_l, "val_loss": val_l, "iou": iou}, step=epoch)
            log.info(f"  Epoch {epoch}/{epochs}  train={train_l:.4f}  val={val_l:.4f}  IoU={iou:.4f}")

            if val_l < best_val_loss:
                best_val_loss = val_l
                torch.save(model.state_dict(), MODELS / "unet_best.pt")

            scheduler.step()

        mlflow.log_metric("best_val_loss", best_val_loss)

    torch.save(model.state_dict(), MODELS / "unet_final.pt")
    log.info(f"[DONE] U-Net saved -> models/unet_best.pt  (IoU={iou:.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# B. LSTM Production Forecaster
# ══════════════════════════════════════════════════════════════════════════════

class ProductionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int = 5):
        self.seq_len = seq_len
        self.X, self.y = [], []
        for i in range(len(series) - seq_len):
            self.X.append(series[i: i + seq_len])
            self.y.append(series[i + seq_len])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).unsqueeze(-1),
                torch.tensor([self.y[idx]]))


def train_lstm_per_site(epochs: int = 30, seq_len: int = 5, lr: float = 1e-3):
    log.info("-- Training LSTM production forecasters --")
    feat_path   = PROC / "production_features.csv"
    scaler_path = MODELS / "production_scaler.pkl"

    if not feat_path.exists():
        log.error("production_features.csv not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(feat_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("LSTM_Production")

    lstm_models = {}

    for site, grp in df.groupby("site_id"):
        grp            = grp.sort_values("year")
        series         = grp["production"].values.reshape(-1, 1)
        series_scaled  = scaler.transform(series).flatten()

        if len(series_scaled) < seq_len + 2:
            log.warning(f"  Skipping {site}: insufficient data ({len(series_scaled)} pts)")
            continue

        ds = TimeSeriesDataset(series_scaled, seq_len)
        dl = DataLoader(ds, batch_size=4, shuffle=True)

        model     = ProductionLSTM().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses    = []

        with mlflow.start_run(run_name=f"lstm_{site}", nested=True):
            mlflow.log_params({"site": site, "epochs": epochs, "seq_len": seq_len})
            for epoch in range(1, epochs + 1):
                model.train()
                losses = []
                for X_batch, y_batch in dl:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                if epoch % 10 == 0:
                    mlflow.log_metric("train_loss", np.mean(losses), step=epoch)

        torch.save(model.state_dict(), MODELS / f"lstm_{site}.pt")
        lstm_models[site] = model
        log.info(f"  [OK] LSTM trained for {site}  (loss={np.mean(losses):.5f})")

    log.info("[DONE] LSTM models saved -> models/lstm_*.pt")
    return lstm_models


# ══════════════════════════════════════════════════════════════════════════════
# C. Prophet Forecaster (fallback / comparison)
# ══════════════════════════════════════════════════════════════════════════════

def train_prophet_forecasters():
    log.info("-- Training Prophet forecasters --")
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("Prophet not installed - skipping. Run: pip install prophet")
        return

    prod_path      = PROC / "production_timeseries.csv"
    df             = pd.read_csv(prod_path)
    prophet_models = {}

    for site, grp in df.groupby("site_id"):
        grp      = grp.sort_values("year").copy()
        grp["ds"] = pd.to_datetime(grp["year"].astype(str) + "-01-01")
        grp["y"]  = grp["production"]
        train    = grp[["ds", "y"]].dropna()

        if len(train) < 5:
            continue

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        m.fit(train, iter=200)
        with open(MODELS / f"prophet_{site}.pkl", "wb") as f:
            pickle.dump(m, f)
        prophet_models[site] = m
        log.info(f"  [OK] Prophet trained for {site}")

    log.info("[DONE] Prophet models saved -> models/prophet_*.pkl")
    return prophet_models


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Step 3: Model training")
    log.info("=" * 60)

    train_unet(epochs=8)
    train_lstm_per_site(epochs=30)
    train_prophet_forecasters()

    log.info("=" * 60)
    log.info("[DONE] All models trained and saved to models/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()