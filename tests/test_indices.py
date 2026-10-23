import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from indices import bare_mask, bsi, clear_mask, dn_to_reflectance, mndwi, ndvi  # noqa: E402


def test_reflectance_before_baseline_4():
    out = dn_to_reflectance(np.array([2500, 0]), baseline=2.12)
    assert out[0] == pytest.approx(0.25)
    assert np.isnan(out[1])


def test_reflectance_after_baseline_4_removes_offset():
    out = dn_to_reflectance(np.array([3500, 500, 0]), baseline=5.11)
    assert out[0] == pytest.approx(0.25)
    assert out[1] == 0.0  # below the offset clips to zero
    assert np.isnan(out[2])


def test_ndvi_known_values():
    nir = np.array([0.5, 0.2, 0.1])
    red = np.array([0.1, 0.2, 0.3])
    assert ndvi(nir, red) == pytest.approx([0.4 / 0.6, 0.0, -0.5])


def test_ndvi_zero_sum_and_nan_give_nan():
    out = ndvi(np.array([0.0, np.nan]), np.array([0.0, 0.1]))
    assert np.isnan(out).all()


def test_bsi_known_value():
    # ((0.30 + 0.20) - (0.25 + 0.05)) / (0.50 + 0.30) = 0.25
    assert bsi(0.30, 0.20, 0.25, 0.05) == pytest.approx(0.25)


def test_mndwi_known_value():
    # (0.10 - 0.02) / (0.10 + 0.02) = 0.667, open water
    assert mndwi(0.10, 0.02) == pytest.approx(0.08 / 0.12)


def test_bare_mask():
    # bare rock, green in summer, water, bright rock just under the water cut, no data
    peak = np.array([0.1, 0.5, 0.1, 0.1, np.nan])
    dry_mndwi = np.array([-0.4, -0.4, 0.7, 0.25, -0.4])
    assert bare_mask(peak, dry_mndwi).tolist() == [True, False, False, True, False]


def test_clear_mask_drops_cloud_shadow_and_nodata():
    scl = np.array([0, 3, 4, 5, 6, 8, 9, 10])
    assert clear_mask(scl).tolist() == [False, False, True, True, True, False, False, False]
