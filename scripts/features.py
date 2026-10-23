"""Step 2: compute NDVI, bare soil index and bare ground per site and year.

Bare ground = highest wet-season NDVI below 0.20 and dry-season MNDWI below
0.30, so not open water (see indices.py). Reads the composites from step 1 and
writes two small CSVs:
  data/processed/site_metrics.csv  one row per site and year
  data/processed/site_change.csv   first year against last year, per site
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PROC, ROOT, composite_path, composite_years, load_sites, read_composite,  # noqa: E402
                    read_wet_ndvi, wet_ndvi_path)
from indices import WATER_MNDWI_MIN, bare_mask, bsi, mndwi, ndvi  # noqa: E402

log = logging.getLogger(__name__)


def index_layers(bands: dict, peak_ndvi: np.ndarray) -> dict:
    """Dry-season NDVI, BSI and MNDWI, wet-season peak NDVI and the bare ground mask."""
    nd = ndvi(bands["B08"], bands["B04"])
    bs = bsi(bands["B11"], bands["B04"], bands["B08"], bands["B02"])
    water = mndwi(bands["B03"], bands["B11"])
    valid = ~np.isnan(nd) & ~np.isnan(water) & ~np.isnan(peak_ndvi)
    return {"ndvi": nd, "bsi": bs, "mndwi": water, "peak_ndvi": peak_ndvi,
            "bare": bare_mask(peak_ndvi, water), "valid": valid}


def load_layers(site_id: str, year: int) -> tuple[dict, dict, dict]:
    """Read both composites for a site and year; return (bands, layers, tags)."""
    bands, tags = read_composite(composite_path(site_id, year))
    peak, wet_tags = read_wet_ndvi(wet_ndvi_path(site_id, year))
    tags["wet_scenes"] = wet_tags.get("scenes", "")
    return bands, index_layers(bands, peak), tags


def year_metrics(layers: dict, pixel_km2: float) -> dict:
    valid = layers["valid"]
    n_valid = int(valid.sum())
    bare = layers["bare"] & valid
    return {
        "valid_frac": round(n_valid / valid.size, 4),
        "ndvi_mean": round(float(np.nanmean(layers["ndvi"])), 4),
        "peak_ndvi_mean": round(float(np.nanmean(layers["peak_ndvi"])), 4),
        "bsi_mean": round(float(np.nanmean(layers["bsi"])), 4),
        "bare_frac": round(float(bare.sum() / n_valid), 4),
        "bare_km2": round(float(bare.sum() * pixel_km2), 3),
        "water_km2": round(float((layers["mndwi"] >= WATER_MNDWI_MIN).sum() * pixel_km2), 3),
        # First rule tried, dry season only; kept to show why the wet-season check was added.
        "bare_km2_dry_rule": round(float(((layers["ndvi"] < 0.15) & (layers["bsi"] > 0.10) & valid).sum()
                                         * pixel_km2), 3),
    }


def bare_change(first: dict, last: dict, pixel_km2: float) -> dict:
    """Area that turned bare, and area that stopped being bare, between two years."""
    both = first["valid"] & last["valid"]
    gained = last["bare"] & ~first["bare"] & both
    lost = first["bare"] & ~last["bare"] & both
    return {
        "bare_gained_km2": round(float(gained.sum() * pixel_km2), 3),
        "bare_lost_km2": round(float(lost.sum() * pixel_km2), 3),
    }


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    rows, change_rows = [], []
    for site in load_sites():
        sid = site["site_id"]
        years = composite_years(sid)
        if not years:
            log.warning(f"No composites for {sid}; run step 1 first")
            continue
        first_layers = last_layers = None
        for year in years:
            _, layers, tags = load_layers(sid, year)
            pixel_km2 = tags["pixel_area_m2"] / 1e6
            rows.append({
                "site_id": sid, "year": year,
                "dry_scenes": len(tags["scenes"].split(",")),
                "wet_scenes": len(tags["wet_scenes"].split(",")),
                "gap_filled_frac": float(tags.get("filled_frac", "nan")),
                **year_metrics(layers, pixel_km2),
            })
            if year == years[0]:
                first_layers = layers
            if year == years[-1]:
                last_layers = layers

        site_rows = pd.DataFrame([r for r in rows if r["site_id"] == sid])
        f, l = site_rows.iloc[0], site_rows.iloc[-1]
        change_rows.append({
            "site_id": sid, "first_year": int(f["year"]), "last_year": int(l["year"]),
            "chip_km2": round(float(site["chip_km"]) ** 2, 1),
            "bare_km2_first": f["bare_km2"], "bare_km2_last": l["bare_km2"],
            "bare_km2_change": round(l["bare_km2"] - f["bare_km2"], 3),
            **bare_change(first_layers, last_layers, pixel_km2),
            "ndvi_mean_first": f["ndvi_mean"], "ndvi_mean_last": l["ndvi_mean"],
            "peak_ndvi_mean_first": f["peak_ndvi_mean"], "peak_ndvi_mean_last": l["peak_ndvi_mean"],
        })
        log.info(f"[OK] {sid}: {len(years)} years, bare ground {f['bare_km2']:.2f} -> {l['bare_km2']:.2f} km2")

    pd.DataFrame(rows).to_csv(PROC / "site_metrics.csv", index=False)
    pd.DataFrame(change_rows).to_csv(PROC / "site_change.csv", index=False)
    log.info(f"[OK] {len(rows)} site-years -> {(PROC / 'site_metrics.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
