import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from download_and_process import chip_grid, dry_composite, wet_max_ndvi  # noqa: E402


def test_chip_grid_is_utm_south_and_10m():
    crs, transform, size = chip_grid(27.5289, -23.6717, chip_km=8)
    assert crs.to_epsg() == 32735
    assert size == 800
    assert transform.a == 10 and transform.e == -10


def test_dry_composite_median_and_gap_fallback():
    # three scenes, one band, two pixels; pixel 2 is masked in every scene
    masked = np.array([[[[0.1, np.nan]]], [[[0.3, np.nan]]], [[[0.2, np.nan]]]], dtype=np.float32)
    unmasked = np.array([[[[0.1, 0.5]]], [[[0.3, 0.7]]], [[[0.2, 0.6]]]], dtype=np.float32)
    comp, gap_frac = dry_composite(masked, unmasked)
    assert comp[0, 0].tolist() == pytest.approx([0.2, 0.6])
    assert gap_frac == 0.5


def test_wet_max_ndvi_takes_greenest_clear_scene():
    # bands are (B04, B08); scene 2 is cloud-masked on the first pixel
    masked = np.array([
        [[[0.10, 0.10]], [[0.30, 0.20]]],          # NDVI 0.5, 0.333
        [[[np.nan, 0.05]], [[np.nan, 0.15]]],      # masked, 0.5
    ], dtype=np.float32)
    peak, nodata_frac = wet_max_ndvi(masked)
    assert peak[0].tolist() == pytest.approx([0.5, 0.5])
    assert nodata_frac == 0.0
