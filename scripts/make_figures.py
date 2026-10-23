"""Step 3: draw the README figures from the composites and the metrics CSV.

  docs/<site_id>_change.png  true colour first and last year, and a bare ground change map
  docs/ndvi_bare_trends.png  mean dry-season NDVI and bare ground share per year, all sites
"""
import io
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS, PROC, composite_years, load_sites, true_colour  # noqa: E402
from features import load_layers  # noqa: E402

log = logging.getLogger(__name__)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GAINED = np.array([0xEB, 0x68, 0x34]) / 255  # bare in the last year only
LOST = np.array([0x2A, 0x78, 0xD6]) / 255    # bare in the first year only
STABLE = np.array([0xC3, 0xC2, 0xB7]) / 255  # bare in both years


def save_small_png(fig, path: Path, colours: int = 256):
    """Save a figure as a palette PNG to keep README images small."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").quantize(colors=colours, method=Image.Quantize.MEDIANCUT)
    img.save(path, optimize=True)
    log.info(f"[OK] {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def change_map(rgb_last: np.ndarray, first: dict, last: dict) -> np.ndarray:
    grey = rgb_last.mean(axis=-1, keepdims=True) * 0.6 + 0.2
    out = np.repeat(grey, 3, axis=-1)
    both = first["valid"] & last["valid"]
    out[first["bare"] & last["bare"] & both] = STABLE
    out[last["bare"] & ~first["bare"] & both] = GAINED
    out[first["bare"] & ~last["bare"] & both] = LOST
    return out


def site_figure(site: dict, metrics: pd.DataFrame):
    sid = site["site_id"]
    years = composite_years(sid)
    y0, y1 = years[0], years[-1]
    b0, l0, _ = load_layers(sid, y0)
    b1, l1, _ = load_layers(sid, y1)
    rgb0, rgb1 = true_colour(b0), true_colour(b1)
    m = metrics[metrics.site_id == sid].set_index("year")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), facecolor=SURFACE)
    panels = [
        (rgb0, f"{y0} dry season, true colour"),
        (rgb1, f"{y1} dry season, true colour"),
        (change_map(rgb1, l0, l1), f"Bare ground, {y0} vs {y1}"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[2].legend(
        handles=[Patch(color=GAINED, label=f"bare in {y1} only"),
                 Patch(color=LOST, label=f"bare in {y0} only"),
                 Patch(color=STABLE, label="bare both years")],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9,
        handlelength=1.2, columnspacing=1.0,
    )
    km = site["chip_km"]
    fig.suptitle(
        f"{site['name']} ({site['mineral']}, {site['kind']}). Chip {km} x {km} km. "
        f"Bare ground {m.loc[y0, 'bare_km2']:.1f} km² in {y0}, {m.loc[y1, 'bare_km2']:.1f} km² in {y1}.",
        fontsize=11, color=INK_2, x=0.01, ha="left",
    )
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data, via Microsoft Planetary Computer.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_small_png(fig, DOCS / f"{sid}_change.png")


def trends_figure(sites: list[dict], metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    specs = [("ndvi_mean", "Mean dry-season NDVI inside the chip", 1.0),
             ("bare_frac", "Bare ground, % of chip", 100.0)]
    for ax, (col, title, scale) in zip(axes, specs):
        for colour, site in zip(SERIES, sites):
            d = metrics[metrics.site_id == site["site_id"]]
            ax.plot(d["year"], d[col] * scale, color=colour, linewidth=2, marker="o", markersize=5,
                    label=site["name"])
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.set_xticks(sorted(metrics["year"].unique()))
        ax.tick_params(axis="x", labelrotation=0)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 60)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Sentinel-2 L2A, 2017 to 2026. Bare ground = never green in January to April "
                 "and not open water in June to September.", fontsize=11, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save_small_png(fig, DOCS / "ndvi_bare_trends.png")


def main():
    DOCS.mkdir(exist_ok=True)
    metrics = pd.read_csv(PROC / "site_metrics.csv")
    sites = load_sites()
    for site in sites:
        if composite_years(site["site_id"]):
            site_figure(site, metrics)
    trends_figure(sites, metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
