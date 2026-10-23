"""Spectral index maths for Sentinel-2 L2A surface reflectance.

All functions take float arrays of reflectance (0 to 1) and return float32
arrays. NaN marks pixels with no valid data and passes through untouched.
"""
import numpy as np

# Sentinel-2 L2A products made with processing baseline 04.00 or later
# (from 25 January 2022) add 1000 to every digital number, so reflectance
# is (DN - 1000) / 10000. Older products use DN / 10000.
QUANT = 10_000.0
BOA_OFFSET_FROM_BASELINE = 4.0
BOA_OFFSET = -1000.0

# Scene Classification Layer (SCL) classes treated as unusable:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium prob,
# 9 cloud high prob, 10 thin cirrus, 11 snow/ice.
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)

# Bare ground rule: the pixel never turned green in the wet season (highest
# NDVI from January to April stays low) and is not open water in the dry
# season. The usual water cut of MNDWI > 0 also caught bright grey rock in the
# Palabora and Tharisa pits, so the cut is higher. Both values were picked by
# eye on the composites, not calibrated against field data.
BARE_PEAK_NDVI_MAX = 0.20
WATER_MNDWI_MIN = 0.30


def dn_to_reflectance(dn: np.ndarray, baseline: float) -> np.ndarray:
    """Convert L2A digital numbers to reflectance. DN 0 means no data."""
    dn = np.asarray(dn, dtype=np.float32)
    offset = BOA_OFFSET if baseline >= BOA_OFFSET_FROM_BASELINE else 0.0
    refl = (dn + offset) / QUANT
    refl[dn == 0] = np.nan
    return np.clip(refl, 0.0, None)


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where the sum is zero."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(total != 0, (a - b) / total, np.nan)
    return out.astype(np.float32)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index, Sentinel-2 B08 and B04."""
    return normalised_difference(nir, red)


def bsi(swir1: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Bare Soil Index: ((SWIR1 + R) - (NIR + B)) / ((SWIR1 + R) + (NIR + B)).

    Sentinel-2 bands B11, B04, B08 and B02.
    """
    return normalised_difference(
        np.asarray(swir1, dtype=np.float32) + red,
        np.asarray(nir, dtype=np.float32) + blue,
    )


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified Normalised Difference Water Index, Sentinel-2 B03 and B11."""
    return normalised_difference(green, swir1)


def bare_mask(peak_ndvi: np.ndarray, dry_mndwi: np.ndarray,
              ndvi_max: float = BARE_PEAK_NDVI_MAX, water_min: float = WATER_MNDWI_MIN) -> np.ndarray:
    """True where a pixel never turned green and is not water. NaN inputs give False."""
    with np.errstate(invalid="ignore"):
        return (peak_ndvi < ndvi_max) & (dry_mndwi < water_min)


def clear_mask(scl: np.ndarray) -> np.ndarray:
    """True where the SCL class is usable (not cloud, shadow, snow or no data)."""
    return ~np.isin(scl, SCL_BAD)
