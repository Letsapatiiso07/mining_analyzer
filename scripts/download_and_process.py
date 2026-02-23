import os
import sys
import json
import logging
import hashlib
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, box

# ── Paths (defined early so log dir exists before logging setup) ──────────────
ROOT    = Path(__file__).resolve().parent.parent
RAW     = ROOT / "data" / "raw"
PROC    = ROOT / "data" / "processed"
IMAGERY = ROOT / "data" / "imagery"
for d in [RAW, PROC, IMAGERY, ROOT / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging (UTF-8 on both stream and file) ───────────────────────────────────
_stream = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)
_stream.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_file = logging.FileHandler(ROOT / "logs" / "pipeline.log", encoding="utf-8")
_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_stream, _file])
log = logging.getLogger(__name__)

# ── South African mine sites of interest ─────────────────────────────────────
SA_SITES = {
    "Witwatersrand_Gold": {
        "lon": 27.8, "lat": -26.2,
        "description": "Johannesburg Gold Fields - largest gold basin",
        "mineral": "Gold",
        "region": "Gauteng",
        "area_km2": 400,
        "production_start": 1886,
    },
    "Rustenburg_Platinum": {
        "lon": 27.24, "lat": -25.67,
        "description": "Rustenburg Platinum Belt - Bushveld Complex",
        "mineral": "Platinum",
        "region": "North West",
        "area_km2": 250,
        "production_start": 1924,
    },
    "Sishen_Iron": {
        "lon": 22.99, "lat": -27.78,
        "description": "Sishen Iron Ore Mine - Northern Cape",
        "mineral": "Iron",
        "region": "Northern Cape",
        "area_km2": 180,
        "production_start": 1953,
    },
    "Palabora_Copper": {
        "lon": 31.13, "lat": -23.98,
        "description": "Palabora Copper Mine - Limpopo",
        "mineral": "Copper",
        "region": "Limpopo",
        "area_km2": 120,
        "production_start": 1964,
    },
    "Grootegeluk_Coal": {
        "lon": 28.07, "lat": -23.91,
        "description": "Grootegeluk Coal Mine - Limpopo",
        "mineral": "Coal",
        "region": "Limpopo",
        "area_km2": 95,
        "production_start": 1980,
    },
}

# ── Production time-series parameters (synthetic but realistic) ───────────────
PRODUCTION_PARAMS = {
    "Gold":     {"base": 250_000, "unit": "kg/yr",  "trend": -0.015, "noise": 0.12},
    "Platinum": {"base": 130_000, "unit": "kg/yr",  "trend": -0.008, "noise": 0.09},
    "Iron":     {"base": 35_000,  "unit": "kt/yr",  "trend":  0.020, "noise": 0.10},
    "Copper":   {"base": 55_000,  "unit": "t/yr",   "trend": -0.010, "noise": 0.11},
    "Coal":     {"base": 18_000,  "unit": "kt/yr",  "trend":  0.005, "noise": 0.08},
}


def safe_download(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file only if it doesn't already exist. Returns True on success."""
    if dest.exists():
        log.info(f"[OK] Already cached: {dest.name}")
        return True
    log.info(f"[->] Downloading {desc or dest.name} ...")
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        log.info(f"    Saved -> {dest}")
        return True
    except Exception as e:
        log.warning(f"    Download failed ({e}). Will generate synthetic fallback.")
        if dest.exists():
            dest.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mine site GeoJSON
# ─────────────────────────────────────────────────────────────────────────────

def build_sites_geojson() -> Path:
    """Build GeoDataFrame of SA mine sites and save as GeoJSON."""
    out = PROC / "sa_mine_sites.geojson"
    if out.exists():
        log.info("[OK] sa_mine_sites.geojson already exists.")
        return out

    records = []
    for name, meta in SA_SITES.items():
        records.append({
            "site_id":  name,
            "geometry": Point(meta["lon"], meta["lat"]),
            **{k: v for k, v in meta.items() if k not in ("lon", "lat")},
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(out, driver="GeoJSON")
    log.info(f"[OK] Wrote {len(gdf)} mine sites -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic production time-series CSV
# ─────────────────────────────────────────────────────────────────────────────

def generate_production_timeseries() -> Path:
    """Generate realistic synthetic production data (1990-2024)."""
    out = PROC / "production_timeseries.csv"
    if out.exists():
        log.info("[OK] production_timeseries.csv already exists.")
        return out

    np.random.seed(42)
    rows  = []
    years = range(1990, 2025)

    for site_name, meta in SA_SITES.items():
        mineral = meta["mineral"]
        p       = PRODUCTION_PARAMS[mineral]
        base    = p["base"]
        trend   = p["trend"]
        noise_std = p["noise"]

        production = base
        for yr in years:
            if yr < meta["production_start"]:
                continue
            cycle      = 0.08 * np.sin(2 * np.pi * (yr - 1990) / 7)
            shock      = np.random.normal(0, noise_std)
            production = max(0, production * (1 + trend + cycle + shock))

            rows.append({
                "site_id":    site_name,
                "year":       yr,
                "mineral":    mineral,
                "unit":       p["unit"],
                "production": round(production, 1),
                "region":     meta["region"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info(f"[OK] Wrote {len(df)} production records -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Sentinel-2 / NDVI rasters
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_raster(site_name: str, meta: dict) -> Path:
    """
    Generate a synthetic 4-band (R,G,B,NIR) GeoTIFF centred on a mine site.
    Models: bare mine pit (low NDVI) at centre, disturbed halo, surrounding vegetation.
    """
    out      = IMAGERY / f"{site_name}_sentinel2.tif"
    ndvi_out = IMAGERY / f"{site_name}_ndvi.tif"
    if out.exists() and ndvi_out.exists():
        log.info(f"[OK] Rasters already exist for {site_name}")
        return out

    np.random.seed(abs(hash(site_name)) % 2**31)

    SIZE     = 256
    RES      = 0.0001
    lon, lat = meta["lon"], meta["lat"]
    half     = SIZE * RES / 2

    xx = np.linspace(lon - half, lon + half, SIZE)
    yy = np.linspace(lat + half, lat - half, SIZE)
    X, Y = np.meshgrid(xx, yy)
    dist = np.sqrt((X - lon)**2 + (Y - lat)**2)

    pit_r  = 0.005 * np.sqrt(meta["area_km2"] / 100)
    halo_r = pit_r * 2.5

    in_pit  = dist < pit_r
    in_halo = (dist >= pit_r) & (dist < halo_r)
    outside = dist >= halo_r

    def band(pit_val, halo_val, veg_val, noise=0.03):
        arr = np.zeros((SIZE, SIZE), dtype=np.float32)
        arr[in_pit]  = np.clip(pit_val  + np.random.normal(0, noise, in_pit.sum()),  0, 1)
        arr[in_halo] = np.clip(halo_val + np.random.normal(0, noise, in_halo.sum()), 0, 1)
        arr[outside] = np.clip(veg_val  + np.random.normal(0, noise, outside.sum()), 0, 1)
        return arr

    red   = band(0.35, 0.25, 0.10, 0.04)
    green = band(0.30, 0.22, 0.12, 0.03)
    blue  = band(0.28, 0.20, 0.08, 0.03)
    nir   = band(0.12, 0.20, 0.45, 0.05)

    transform = from_bounds(lon - half, lat - half, lon + half, lat + half, SIZE, SIZE)
    crs = CRS.from_epsg(4326)

    with rasterio.open(
        out, "w", driver="GTiff",
        height=SIZE, width=SIZE, count=4,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(red,   1)
        dst.write(green, 2)
        dst.write(blue,  3)
        dst.write(nir,   4)
        dst.update_tags(
            site=site_name, mineral=meta["mineral"],
            bands="R,G,B,NIR", source="synthetic_sentinel2",
        )

    ndvi = np.where(
        (nir + red) > 0,
        (nir - red) / (nir + red + 1e-8),
        0.0,
    ).astype(np.float32)

    with rasterio.open(
        ndvi_out, "w", driver="GTiff",
        height=SIZE, width=SIZE, count=1,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(ndvi, 1)
        dst.update_tags(site=site_name, index="NDVI")

    log.info(f"  [OK] Rasters -> {out.name}, {ndvi_out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Segmentation label masks
# ─────────────────────────────────────────────────────────────────────────────

def generate_label_mask(site_name: str, meta: dict) -> Path:
    """Generate pixel-level label raster: 0=background, 1=mine_pit, 2=disturbed, 3=vegetation."""
    out = IMAGERY / f"{site_name}_labels.tif"
    if out.exists():
        return out

    np.random.seed(abs(hash(site_name + "_lbl")) % 2**31)

    SIZE     = 256
    RES      = 0.0001
    lon, lat = meta["lon"], meta["lat"]
    half     = SIZE * RES / 2

    xx = np.linspace(lon - half, lon + half, SIZE)
    yy = np.linspace(lat + half, lat - half, SIZE)
    X, Y = np.meshgrid(xx, yy)
    dist = np.sqrt((X - lon)**2 + (Y - lat)**2)

    pit_r  = 0.005 * np.sqrt(meta["area_km2"] / 100)
    halo_r = pit_r * 2.5

    labels = np.zeros((SIZE, SIZE), dtype=np.uint8)
    labels[dist >= halo_r]                          = 3
    labels[(dist >= pit_r) & (dist < halo_r)]       = 2
    labels[dist < pit_r]                            = 1

    noise = np.random.randint(0, 4, (SIZE, SIZE), dtype=np.uint8)
    edge  = (dist > pit_r * 0.85) & (dist < pit_r * 1.15)
    labels[edge] = noise[edge]
    labels = np.clip(labels, 0, 3).astype(np.uint8)

    transform = from_bounds(lon - half, lat - half, lon + half, lat + half, SIZE, SIZE)
    with rasterio.open(
        out, "w", driver="GTiff",
        height=SIZE, width=SIZE, count=1,
        dtype="uint8", crs=CRS.from_epsg(4326), transform=transform,
    ) as dst:
        dst.write(labels, 1)
        dst.update_tags(classes="0=background,1=mine_pit,2=disturbed,3=vegetation")

    log.info(f"  [OK] Label mask -> {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature CSV for ML
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_features() -> Path:
    """Compute per-site spectral statistics and save as feature CSV."""
    out = PROC / "spectral_features.csv"
    if out.exists():
        log.info("[OK] spectral_features.csv already exists.")
        return out

    rows = []
    for site_name, meta in SA_SITES.items():
        raster_path = IMAGERY / f"{site_name}_sentinel2.tif"
        if not raster_path.exists():
            continue

        with rasterio.open(raster_path) as src:
            red   = src.read(1).flatten()
            green = src.read(2).flatten()
            blue  = src.read(3).flatten()
            nir   = src.read(4).flatten()

        ndvi = (nir - red) / (nir + red + 1e-8)
        evi  = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
        bsi  = ((red + blue) - (nir + green)) / ((red + blue) + (nir + green) + 1e-8)

        rows.append({
            "site_id":       site_name,
            "mineral":       meta["mineral"],
            "area_km2":      meta["area_km2"],
            "ndvi_mean":     float(np.mean(ndvi)),
            "ndvi_std":      float(np.std(ndvi)),
            "ndvi_min":      float(np.min(ndvi)),
            "ndvi_pit_mean": float(np.mean(ndvi[ndvi < -0.1])) if (ndvi < -0.1).any() else 0.0,
            "evi_mean":      float(np.mean(evi)),
            "bsi_mean":      float(np.mean(bsi)),
            "red_mean":      float(np.mean(red)),
            "nir_mean":      float(np.mean(nir)),
            "pit_fraction":  float((ndvi < -0.05).mean()),
            "veg_fraction":  float((ndvi > 0.25).mean()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info(f"[OK] Spectral features -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Step 1: Data download and preprocessing")
    log.info("=" * 60)

    build_sites_geojson()
    generate_production_timeseries()

    log.info("Generating synthetic Sentinel-2 rasters ...")
    for site_name, meta in SA_SITES.items():
        generate_synthetic_raster(site_name, meta)
        generate_label_mask(site_name, meta)

    compute_spectral_features()

    log.info("=" * 60)
    log.info("[DONE] Data pipeline complete. Files in data/processed/ and data/imagery/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()