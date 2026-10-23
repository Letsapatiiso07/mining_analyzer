"""Step 1: build two Sentinel-2 L2A composites per site and year.

For each site, year and season this script:
  1. searches the Microsoft Planetary Computer STAC API for Sentinel-2 L2A scenes,
  2. reads only the chip around the site from the cloud-optimised GeoTIFFs,
  3. ranks scenes by clear-sky fraction inside the chip (from the SCL band),
  4. masks cloud, shadow and snow in the best scenes and combines them:
       dry season (June to September): per-pixel median of B02, B03, B04, B08, B11,
         written as data/imagery/<site>/<year>.tif (reflectance x 10000)
       wet season (January to April): per-pixel highest NDVI,
         written as data/imagery/<site>/<year>_wet_max_ndvi.tif (NDVI x 10000)

It also writes data/processed/scenes.csv, listing every scene used.
"""
import argparse
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from pystac_client.stac_api_io import StacApiIO
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform
from urllib3 import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BANDS, PROC, ROOT, composite_path, load_sites, wet_ndvi_path  # noqa: E402
from indices import clear_mask, dn_to_reflectance, ndvi  # noqa: E402

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
YEARS = list(range(2017, 2027))
SEASONS = {
    # Winter: little cloud and little green grass. All five sites get most of their rain in summer.
    "dry": {"dates": ("06-01", "09-30"), "bands": BANDS, "max_tile_cloud": 20, "candidates": 12,
            "min_chip_clear": 0.8},
    # Summer: grass and bush are green, so ground that stays bare here is not just dry veld.
    "wet": {"dates": ("01-01", "04-30"), "bands": ["B04", "B08"], "max_tile_cloud": 60, "candidates": 15,
            "min_chip_clear": 0.5},
}
# max_tile_cloud: % cloud over the whole 100 km tile (item metadata)
# candidates: least cloudy scenes whose SCL band we read to rank by clear sky in the chip
# min_chip_clear: skip scenes with less clear sky than this inside the chip
MAX_SCENES = 5        # scenes that go into each composite
RES = 10              # metres

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

log = logging.getLogger(__name__)


def chip_grid(lon: float, lat: float, chip_km: float):
    """Square 10 m grid centred on the site, in its UTM zone (southern hemisphere)."""
    zone = int((lon + 180) // 6) + 1
    crs = CRS.from_epsg(32700 + zone)
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    size = int(round(chip_km * 1000 / RES))
    half = size * RES / 2
    x0 = round((xs[0] - half) / RES) * RES
    y0 = round((ys[0] + half) / RES) * RES
    return crs, from_origin(x0, y0, RES, RES), size


def grid_bbox_lonlat(crs, transform, size) -> list[float]:
    x0, y0 = transform.c, transform.f
    x1, y1 = x0 + size * RES, y0 - size * RES
    lons, lats = warp_transform(crs, "EPSG:4326", [x0, x1, x0, x1], [y0, y0, y1, y1])
    return [min(lons), min(lats), max(lons), max(lats)]


def read_on_grid(href: str, crs, transform, size: int, resampling) -> np.ndarray:
    with rasterio.open(href) as src, WarpedVRT(
        src, crs=crs, transform=transform, width=size, height=size,
        resampling=resampling, src_nodata=0, nodata=0,
    ) as vrt:
        return vrt.read(1)


def search_candidates(catalog, bbox, year: int, season: dict) -> list:
    first, last = season["dates"]
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{year}-{first}/{year}-{last}",
        query={"eo:cloud_cover": {"lt": season["max_tile_cloud"]}},
    )
    items = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])
    return items[:season["candidates"]]


def read_scenes(catalog, site: dict, year: int, season_name: str, pool):
    """Read the clearest scenes for one site, year and season.

    Returns (masked, unmasked, scene_rows, grid). masked and unmasked have shape
    (scenes, bands, rows, cols) in reflectance; masked has NaN under cloud.
    """
    season = SEASONS[season_name]
    bands = season["bands"]
    crs, transform, size = chip_grid(site["lon"], site["lat"], site["chip_km"])
    bbox = grid_bbox_lonlat(crs, transform, size)
    items = [planetary_computer.sign(i) for i in search_candidates(catalog, bbox, year, season)]
    if not items:
        log.warning(f"  {site['site_id']} {year} {season_name}: no scenes found")
        return None

    # Rank by clear sky inside the chip. Overlapping tiles from the same day
    # carry the same pixels, so keep only the best one per day.
    scls = list(pool.map(
        lambda it: read_on_grid(it.assets["SCL"].href, crs, transform, size, Resampling.nearest),
        items,
    ))
    best_per_day = {}
    for item, scl in zip(items, scls):
        clear = float(clear_mask(scl).mean())
        day = item.datetime.date()
        if clear >= season["min_chip_clear"] and clear > best_per_day.get(day, (None, None, -1.0))[2]:
            best_per_day[day] = (item, scl, clear)
    chosen = sorted(best_per_day.values(), key=lambda t: -t[2])[:MAX_SCENES]
    if not chosen:
        log.warning(f"  {site['site_id']} {year} {season_name}: no scene clear enough in the chip")
        return None

    masked = np.full((len(chosen), len(bands), size, size), np.nan, dtype=np.float32)
    unmasked = np.full_like(masked, np.nan)
    scene_rows = []
    for i, (item, scl, clear) in enumerate(chosen):
        baseline = float(item.properties.get("s2:processing_baseline") or 0)
        ok = clear_mask(scl)
        dns = list(pool.map(
            lambda b: read_on_grid(item.assets[b].href, crs, transform, size, Resampling.bilinear),
            bands,
        ))
        for j, dn in enumerate(dns):
            refl = dn_to_reflectance(dn, baseline)
            unmasked[i, j] = refl
            masked[i, j] = np.where(ok, refl, np.nan)
        scene_rows.append({
            "site_id": site["site_id"], "year": year, "season": season_name, "item_id": item.id,
            "date": item.datetime.date().isoformat(),
            "tile_cloud_pct": item.properties["eo:cloud_cover"],
            "chip_clear_frac": round(clear, 4),
            "processing_baseline": item.properties.get("s2:processing_baseline"),
        })
    return masked, unmasked, scene_rows, (crs, transform)


def dry_composite(masked: np.ndarray, unmasked: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-pixel median of the clear scenes.

    Pixels that every scene flags as cloud or shadow (often bright pit floors
    or tailings) fall back to the unmasked median of the same scenes. Returns
    the composite and the share of such pixels.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        comp = np.nanmedian(masked, axis=0)
        gaps = np.isnan(comp)
        if gaps.any():
            comp = np.where(gaps, np.nanmedian(unmasked, axis=0), comp)
    return comp, float(gaps.any(axis=0).mean())


def wet_max_ndvi(masked: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-pixel highest NDVI over the scenes (bands B04, B08).

    Cloud has low NDVI, so the maximum is little affected by cloud the mask
    misses. Returns the NDVI array and the share of pixels with no clear scene.
    """
    scene_ndvi = ndvi(masked[:, 1], masked[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(scene_ndvi, axis=0)
    return peak, float(np.isnan(peak).mean())


def write_raster(path: Path, data: np.ndarray, grid, nodata, band_names: list[str], tags: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    crs, transform = grid
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype=data.dtype.name, crs=crs, transform=transform,
        nodata=nodata, compress="deflate", predictor=2, tiled=True,
    ) as dst:
        dst.write(data)
        for j, name in enumerate(band_names, start=1):
            dst.set_band_description(j, name)
        dst.update_tags(**tags)


def build(catalog, site: dict, year: int, season_name: str, out: Path, pool) -> list[dict] | None:
    result = read_scenes(catalog, site, year, season_name, pool)
    if result is None:
        return None
    masked, unmasked, scene_rows, grid = result
    tags = {
        "site_id": site["site_id"], "year": str(year),
        "season": " to ".join(SEASONS[season_name]["dates"]),
        "scenes": ",".join(r["item_id"] for r in scene_rows),
    }
    if season_name == "dry":
        comp, gap_frac = dry_composite(masked, unmasked)
        scaled = np.clip(np.round(np.nan_to_num(comp, nan=0.0) * 10_000), 0, 65_535).astype(np.uint16)
        scaled[~np.isnan(comp) & (scaled == 0)] = 1  # keep 0 for no data
        write_raster(out, scaled, grid, 0, BANDS,
                     {**tags, "scale": "reflectance x 10000", "filled_frac": f"{gap_frac:.4f}"})
    else:
        peak, gap_frac = wet_max_ndvi(masked)
        scaled = np.where(np.isnan(peak), -32768, np.round(peak * 10_000)).astype(np.int16)[None]
        write_raster(out, scaled, grid, -32768, ["max_ndvi"],
                     {**tags, "scale": "NDVI x 10000", "nodata_frac": f"{gap_frac:.4f}"})
    log.info(f"     {len(scene_rows)} scenes, {gap_frac:.1%} of pixels without a clear view")
    return scene_rows


def save_scene_rows(scenes_csv: Path, rows: list[dict]):
    """Replace the rows for each (site, year, season) in rows and rewrite the CSV."""
    new = pd.DataFrame(rows)
    if scenes_csv.exists():
        old = pd.read_csv(scenes_csv, dtype={"processing_baseline": str})
        done = set(zip(new.site_id, new.year, new.season))
        old = old[[k not in done for k in zip(old.site_id, old.year, old.season)]]
        new = pd.concat([old, new])
    new["processing_baseline"] = new["processing_baseline"].map(lambda v: f"{float(v):05.2f}")
    new.sort_values(["site_id", "year", "season", "date"]).to_csv(scenes_csv, index=False)


def main(years: list[int] | None = None, force: bool = False):
    years = years or YEARS
    PROC.mkdir(parents=True, exist_ok=True)
    scenes_csv = PROC / "scenes.csv"

    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
    catalog = pystac_client.Client.open(STAC_URL, stac_io=StacApiIO(max_retries=retry))
    failed = []
    with rasterio.Env(**GDAL_ENV), ThreadPoolExecutor(max_workers=4) as pool:
        for site in load_sites():
            for year in years:
                for season_name, out in [("dry", composite_path(site["site_id"], year)),
                                         ("wet", wet_ndvi_path(site["site_id"], year))]:
                    if out.exists() and not force:
                        log.info(f"[OK] {out.relative_to(ROOT)} exists")
                        continue
                    log.info(f"[->] {site['site_id']} {year} {season_name} season")
                    try:
                        scene_rows = build(catalog, site, year, season_name, out, pool)
                    except Exception as e:  # network errors: log, carry on, retry on the next run
                        log.error(f"     failed: {e}")
                        failed.append(str(out.relative_to(ROOT)))
                        continue
                    if scene_rows:
                        save_scene_rows(scenes_csv, scene_rows)
    if failed:
        log.error(f"{len(failed)} composites failed; run step 1 again to retry them")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build dry- and wet-season Sentinel-2 composites per site and year.")
    parser.add_argument("--years", type=int, nargs="+", help="years to fetch (default 2017 to 2026)")
    parser.add_argument("--force", action="store_true", help="rebuild composites that already exist")
    args = parser.parse_args()
    main(args.years, args.force)
