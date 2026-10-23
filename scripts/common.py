"""Paths and helpers shared by the pipeline steps and the Streamlit app."""
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
SITES_FILE = ROOT / "data" / "sa_mine_sites.geojson"
IMAGERY = ROOT / "data" / "imagery"
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

BANDS = ["B02", "B03", "B04", "B08", "B11"]  # blue, green, red, NIR, SWIR1
RGB_MAX = 0.30  # reflectance shown as full brightness in true-colour images


def load_sites() -> list[dict]:
    with open(SITES_FILE, encoding="utf-8") as f:
        features = json.load(f)["features"]
    sites = []
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"]
        sites.append({**feat["properties"], "lon": lon, "lat": lat})
    return sites


def composite_path(site_id: str, year: int) -> Path:
    """Dry-season five-band composite."""
    return IMAGERY / site_id / f"{year}.tif"


def wet_ndvi_path(site_id: str, year: int) -> Path:
    """Wet-season highest NDVI."""
    return IMAGERY / site_id / f"{year}_wet_max_ndvi.tif"


def composite_years(site_id: str) -> list[int]:
    """Years that have both composites."""
    years = (int(p.stem) for p in (IMAGERY / site_id).glob("[0-9][0-9][0-9][0-9].tif"))
    return sorted(y for y in years if wet_ndvi_path(site_id, y).exists())


def read_composite(path: Path) -> tuple[dict, dict]:
    """Return ({band name: reflectance array with NaN for no data}, tags)."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        tags = src.tags()
        tags["pixel_area_m2"] = abs(src.res[0] * src.res[1])
    data[data == 0] = np.nan
    data /= 10_000.0
    return dict(zip(BANDS, data)), tags


def read_wet_ndvi(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        tags = src.tags()
    data[data == -32768] = np.nan
    return data / 10_000.0, tags


def true_colour(bands: dict) -> np.ndarray:
    """RGB array in 0..1 with one fixed stretch, so years compare fairly."""
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    rgb = np.clip(np.nan_to_num(rgb) / RGB_MAX, 0, 1)
    return rgb ** (1 / 1.5)
