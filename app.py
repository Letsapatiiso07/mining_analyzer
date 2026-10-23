"""Streamlit viewer for the Sentinel-2 composites and index metrics.

Run the pipeline first (python run_pipeline.py), then: streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import PROC, composite_years, load_sites, true_colour  # noqa: E402
from features import load_layers  # noqa: E402
from indices import BARE_PEAK_NDVI_MAX, WATER_MNDWI_MIN  # noqa: E402
from make_figures import change_map  # noqa: E402

st.set_page_config(page_title="SA Mine Site Change", layout="wide")


@st.cache_data
def load_tables():
    def read(name):
        p = PROC / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return read("site_metrics.csv"), read("site_change.csv"), read("scenes.csv")


@st.cache_data
def site_year(site_id: str, year: int):
    bands, layers, _ = load_layers(site_id, year)
    return true_colour(bands), layers


def ndvi_image(nd: np.ndarray) -> np.ndarray:
    """Brown (low) to green (high) for NDVI 0 to 0.8."""
    t = np.clip(np.nan_to_num(nd, nan=0.0) / 0.8, 0, 1)[..., None]
    low, high = np.array([0.55, 0.42, 0.30]), np.array([0.10, 0.50, 0.18])
    return low * (1 - t) + high * t


sites = {s["site_id"]: s for s in load_sites()}
metrics, change, scenes = load_tables()

with st.sidebar:
    st.header("Site")
    ids = list(sites)
    start = st.query_params.get("site")  # e.g. ?site=Tharisa_Platinum
    sid = st.selectbox("Mine site", ids, index=ids.index(start) if start in sites else 0,
                       format_func=lambda s: sites[s]["name"])
    site = sites[sid]
    years = composite_years(sid)
    if years:
        y0, y1 = st.select_slider("Compare years", options=years, value=(years[0], years[-1]))
    st.markdown(
        f"**{site['mineral']}**, {site['kind']}  \n"
        f"Operator: {site['operator']}  \n"
        f"{site['region']}, {abs(site['lat']):.4f} S {site['lon']:.4f} E  \n"
        f"Chip: {site['chip_km']} x {site['chip_km']} km  \n"
        f"Location source: [{site['coord_source']}]({site['coord_url']})"
    )
    st.caption(site["note"])

st.title("South African mine sites from Sentinel-2")
st.caption("Two composites per year at 10 m: dry season (June to September) median for the images, "
           "wet season (January to April) highest NDVI for the bare ground rule. "
           "Contains modified Copernicus Sentinel data, via Microsoft Planetary Computer.")

tab_img, tab_trend, tab_all, tab_method = st.tabs(["Imagery", "Trends", "All sites", "Method and limits"])

with tab_img:
    if not years:
        st.info("No composites for this site yet. Run `python run_pipeline.py` first.")
    else:
        (rgb0, l0), (rgb1, l1) = site_year(sid, y0), site_year(sid, y1)
        c1, c2, c3, c4 = st.columns(4)
        c1.image(rgb0, caption=f"{y0} dry season, true colour", width="stretch", clamp=True)
        c2.image(rgb1, caption=f"{y1} dry season, true colour", width="stretch", clamp=True)
        c3.image(ndvi_image(l1["peak_ndvi"]), caption=f"{y1} wet-season highest NDVI (brown low, green high)",
                 width="stretch", clamp=True)
        c4.image(change_map(rgb1, l0, l1), width="stretch", clamp=True,
                 caption=f"Bare ground: orange = bare in {y1} only, blue = bare in {y0} only, "
                         f"grey = bare both years")
        if not metrics.empty:
            m = metrics[metrics.site_id == sid].set_index("year")
            a, b, c = st.columns(3)
            a.metric(f"Bare ground {y0}", f"{m.loc[y0, 'bare_km2']:.2f} km²")
            b.metric(f"Bare ground {y1}", f"{m.loc[y1, 'bare_km2']:.2f} km²",
                     f"{m.loc[y1, 'bare_km2'] - m.loc[y0, 'bare_km2']:+.2f} km²", delta_color="off")
            c.metric(f"Open water {y1}", f"{m.loc[y1, 'water_km2']:.2f} km²",
                     f"{m.loc[y1, 'water_km2'] - m.loc[y0, 'water_km2']:+.2f} km²", delta_color="off")

with tab_trend:
    m = metrics[metrics.site_id == sid] if not metrics.empty else pd.DataFrame()
    if m.empty:
        st.info("No metrics yet. Run `python run_pipeline.py` first.")
    else:
        a, b, c = st.columns(3)
        first, last = m.iloc[0], m.iloc[-1]
        a.metric(f"Bare ground {int(last.year)}", f"{last.bare_km2:.2f} km²",
                 f"{last.bare_km2 - first.bare_km2:+.2f} km² since {int(first.year)}", delta_color="off")
        b.metric(f"Mean dry-season NDVI {int(last.year)}", f"{last.ndvi_mean:.3f}",
                 f"{last.ndvi_mean - first.ndvi_mean:+.3f} since {int(first.year)}", delta_color="off")
        n = pd.concat([m.dry_scenes, m.wet_scenes])
        c.metric("Scenes per composite", f"{n.min()}" if n.min() == n.max() else f"{n.min()} to {n.max()}")
        for col, title, unit in [("bare_km2", "Bare ground inside the chip", "km²"),
                                 ("ndvi_mean", "Mean dry-season NDVI inside the chip", "NDVI"),
                                 ("water_km2", "Open water inside the chip", "km²")]:
            fig = go.Figure(go.Scatter(x=m.year, y=m[col], mode="lines+markers",
                                       line=dict(color="#2a78d6", width=2), marker=dict(size=8),
                                       hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
            fig.update_layout(title=title, yaxis_title=unit, height=320, yaxis=dict(rangemode="tozero"),
                              margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(dtick=1))
            st.plotly_chart(fig, width="stretch")
        st.dataframe(m.drop(columns="site_id"), hide_index=True, width="stretch")
        with st.expander("Scenes used"):
            if not scenes.empty:
                st.dataframe(scenes[scenes.site_id == sid].drop(columns="site_id"), hide_index=True,
                             width="stretch")

with tab_all:
    st.map(pd.DataFrame([{"lat": s["lat"], "lon": s["lon"]} for s in sites.values()]), zoom=4)
    if not change.empty:
        table = change.assign(site=change.site_id.map(lambda s: sites[s]["name"]))
        st.dataframe(table.set_index("site").drop(columns="site_id"), width="stretch")

with tab_method:
    st.markdown(f"""
- **Composites.** For each year the pipeline takes up to five Sentinel-2 L2A scenes with the most
  clear sky inside the chip from two seasons, and masks cloud, cloud shadow and snow with the
  Scene Classification Layer. Dry season (June to September): per-pixel median of five bands.
  Wet season (January to April): per-pixel highest NDVI.
- **NDVI** = (B08 - B04) / (B08 + B04). **MNDWI** = (B03 - B11) / (B03 + B11).
  **BSI** = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02)), reported but not used in the rule.
- **Bare ground** = wet-season highest NDVI below {BARE_PEAK_NDVI_MAX} (it never turned green) and
  dry-season MNDWI below {WATER_MNDWI_MIN} (it is not open water). The thresholds were picked by
  eye, not calibrated against field data.
- **Limits.** Bare ground is not a mine boundary: it also counts roads, roofs, ploughed fields and
  sparse natural cover. A poor rainy season leaves more veld below the NDVI cut. Water in pits and
  on tailings dams is not counted as bare, so a pond that grows or shrinks changes the bare area
  without any mining.
""")
