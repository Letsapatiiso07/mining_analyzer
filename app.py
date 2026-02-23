import io
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import folium
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" SA Mining Site Analyzer",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT    = Path(__file__).resolve().parent
PROC    = ROOT / "data" / "processed"
IMAGERY = ROOT / "data" / "imagery"
MODELS  = ROOT / "models"
PREDS   = PROC / "predictions"

# ── Site metadata ─────────────────────────────────────────────────────────────
SA_SITES = {
    "Witwatersrand_Gold": {
        "lon": 27.8, "lat": -26.2,
        "description": "Johannesburg Gold Fields – largest gold basin in the world",
        "mineral": "Gold", "region": "Gauteng", "area_km2": 400,
    },
    "Rustenburg_Platinum": {
        "lon": 27.24, "lat": -25.67,
        "description": "Rustenburg Platinum Belt – Bushveld Igneous Complex",
        "mineral": "Platinum", "region": "North West", "area_km2": 250,
    },
    "Sishen_Iron": {
        "lon": 22.99, "lat": -27.78,
        "description": "Sishen Iron Ore Mine – largest open-cast iron mine in Africa",
        "mineral": "Iron", "region": "Northern Cape", "area_km2": 180,
    },
    "Palabora_Copper": {
        "lon": 31.13, "lat": -23.98,
        "description": "Palabora Copper Mine – block-cave and historic open pit",
        "mineral": "Copper", "region": "Limpopo", "area_km2": 120,
    },
    "Grootegeluk_Coal": {
        "lon": 28.07, "lat": -23.91,
        "description": "Grootegeluk Coal Mine – Waterberg coalfield, Limpopo",
        "mineral": "Coal", "region": "Limpopo", "area_km2": 95,
    },
}

MINERAL_ICONS = {"Gold": "🥇", "Platinum": "💎", "Iron": "🔩", "Copper": "🟠", "Coal": "⚫"}
CLASS_COLORS  = ["#808080", "#DC3232", "#FFA500", "#3CB43C"]
CLASS_NAMES   = ["Background", "Mine Pit", "Disturbed Zone", "Vegetation"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    border-radius: 10px; padding: 16px; color: white;
    text-align: center; margin: 4px;
  }
  .metric-card h3 { font-size: 2rem; margin: 0; }
  .metric-card p  { font-size: 0.85rem; opacity: 0.8; margin: 0; }
  .site-badge {
    background: #2d6a9f; color: white; border-radius: 20px;
    padding: 4px 12px; font-size: 0.8rem; display: inline-block;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 6px; padding: 8px 20px;
    background: #1e3a5f; color: #ccc;
  }
  .stTabs [aria-selected="true"] {
    background: #2d6a9f !important; color: white !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_inference_module():
    """Import inference module (cached so models load once)."""
    import sys
    sys.path.insert(0, str(ROOT))
    import inference
    return inference


@st.cache_data
def load_production_data():
    p = PROC / "production_timeseries.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_ndvi_data():
    p = PROC / "ndvi_timeseries.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_pit_metrics():
    p = PROC / "pit_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_sites_geodf():
    p = PROC / "sa_mine_sites.geojson"
    return gpd.read_file(p) if p.exists() else None


def data_ready() -> bool:
    return (PROC / "production_timeseries.csv").exists() and (IMAGERY / "Witwatersrand_Gold_sentinel2.tif").exists()


def raster_to_rgb(site_id: str) -> np.ndarray | None:
    p = IMAGERY / f"{site_id}_sentinel2.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb / (rgb.max() + 1e-6), 0, 1)


def ndvi_to_image(site_id: str) -> np.ndarray | None:
    p = IMAGERY / f"{site_id}_ndvi.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        ndvi = src.read(1).astype(np.float32)
    # Colour map: -1=red (bare/pit)  0=yellow  +1=dark green
    cmap_r = np.clip(1 - (ndvi + 1) / 2, 0, 1)
    cmap_g = np.clip((ndvi + 1) / 2, 0, 1)
    cmap_b = np.zeros_like(ndvi)
    return np.stack([cmap_r, cmap_g, cmap_b], -1)


def seg_to_image(site_id: str) -> np.ndarray | None:
    p = PREDS / f"{site_id}_segmentation.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        seg = src.read(1)
    colors = np.array([[128,128,128],[220,50,50],[255,165,0],[60,180,60]], dtype=np.uint8)
    return colors[seg]


def array_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⛏️ SA Mining Analyzer")
    st.markdown("*Satellite · ML · Environmental*")
    st.divider()

    site_options = list(SA_SITES.keys())
    site_display = {s: f"{MINERAL_ICONS[SA_SITES[s]['mineral']]}  {s.replace('_',' ')}" for s in site_options}

    selected_site = st.selectbox(
        "Select Mine Site",
        options=site_options,
        format_func=lambda s: site_display[s],
        index=0,
    )

    meta = SA_SITES[selected_site]
    st.markdown(f"""
    <div class="metric-card">
      <p>{meta['description']}</p>
      <br/>
      <span class="site-badge">{meta['mineral']}</span>
      <span class="site-badge">{meta['region']}</span>
      <span class="site-badge">{meta['area_km2']} km²</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    seg_alpha = st.slider("Overlay Opacity", 0.2, 1.0, 0.55, 0.05)
    forecast_horizon = st.slider("Forecast Horizon (years)", 1, 10, 5)
    st.divider()

    if not data_ready():
        st.warning("⚠️ Data not found.\nRun the setup pipeline first:\n\n```\npython scripts/01_download_and_process.py\npython scripts/02_features.py\npython scripts/03_train.py\n```")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("⛏️ South African Open-Pit Mining Site Analyzer")
st.markdown(f"**Active site:** {selected_site.replace('_', ' ')}  |  "
            f"**Mineral:** {meta['mineral']}  |  **Region:** {meta['region']}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_map, tab_imagery, tab_seg, tab_forecast, tab_env, tab_upload = st.tabs([
    "🗺️ Site Map", "🛰️ Imagery", "🔍 Segmentation", "📈 Production", "🌿 Environment", "📤 Upload"
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 – Site Map
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:
    st.subheader("Mine Site Locations – South Africa")

    mineral_palette = {
        "Gold": "#FFD700", "Platinum": "#E5E4E2", "Iron": "#8B4513",
        "Copper": "#B87333", "Coal": "#36454F"
    }

    # Folium map
    m = folium.Map(location=[-26.5, 27.0], zoom_start=6,
                   tiles="CartoDB dark_matter")

    for sid, smeta in SA_SITES.items():
        color   = mineral_palette.get(smeta["mineral"], "#888")
        is_sel  = sid == selected_site
        radius  = 18 if is_sel else 12
        opacity = 1.0 if is_sel else 0.7

        folium.CircleMarker(
            location=[smeta["lat"], smeta["lon"]],
            radius=radius, color=color, fill=True, fill_color=color,
            fill_opacity=opacity,
            popup=folium.Popup(
                f"<b>{sid.replace('_', ' ')}</b><br>"
                f"Mineral: {smeta['mineral']}<br>"
                f"Region: {smeta['region']}<br>"
                f"Area: {smeta['area_km2']} km²",
                max_width=200
            ),
            tooltip=sid.replace("_", " "),
        ).add_to(m)

        if is_sel:
            folium.Marker(
                location=[smeta["lat"] + 0.15, smeta["lon"]],
                icon=folium.DivIcon(html=f'<div style="color:{color};font-weight:bold;font-size:11px;">'
                                         f'▶ {sid.replace("_"," ")}</div>'),
            ).add_to(m)

    try:
        from streamlit_folium import st_folium
        st_folium(m, width=None, height=500, returned_objects=[])
    except ImportError:
        # Fallback: render map as HTML
        map_html = m._repr_html_()
        st.components.v1.html(map_html, height=520)

    # Site stats table
    st.subheader("All Sites Overview")
    prod_df = load_production_data()
    rows = []
    for sid, smeta in SA_SITES.items():
        latest = prod_df[prod_df["site_id"] == sid].sort_values("year").iloc[-1:] if not prod_df.empty else pd.DataFrame()
        latest_prod = f"{latest['production'].iloc[0]:,.0f} {latest['unit'].iloc[0]}" if not latest.empty else "—"
        rows.append({
            "Site": sid.replace("_", " "),
            "Mineral": smeta["mineral"],
            "Region":  smeta["region"],
            "Area (km²)": smeta["area_km2"],
            "Latest Production": latest_prod,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 – Imagery
# ══════════════════════════════════════════════════════════════════════════════

with tab_imagery:
    st.subheader(f"Satellite Imagery – {selected_site.replace('_', ' ')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**RGB Composite** (Bands R-G-B)")
        rgb = raster_to_rgb(selected_site)
        if rgb is not None:
            st.image(array_to_pil(rgb), use_column_width=True)
        else:
            st.info("Run the data pipeline to generate imagery.")

    with col2:
        st.markdown("**NDVI Map** (Red=bare, Green=vegetation)")
        ndvi_img = ndvi_to_image(selected_site)
        if ndvi_img is not None:
            st.image(array_to_pil(ndvi_img), use_column_width=True)
        else:
            st.info("NDVI raster not found.")

    with col3:
        st.markdown("**Segmentation Mask** (pre-computed)")
        seg_img = seg_to_image(selected_site)
        if seg_img is not None:
            st.image(array_to_pil(seg_img), use_column_width=True)
        else:
            st.info("Run inference pipeline to generate segmentation.")

    # Legend
    st.markdown("**Legend:**")
    leg_cols = st.columns(4)
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        with leg_cols[i]:
            st.markdown(
                f'<div style="background:{color};color:white;padding:6px 10px;'
                f'border-radius:6px;text-align:center;font-size:0.85rem">{name}</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 – Segmentation
# ══════════════════════════════════════════════════════════════════════════════

with tab_seg:
    st.subheader(f"Semantic Segmentation – {selected_site.replace('_', ' ')}")

    col_img, col_stats = st.columns([2, 1])

    with col_img:
        rgb = raster_to_rgb(selected_site)
        seg = seg_to_image(selected_site)

        if rgb is not None and seg is not None:
            # Blend overlay
            overlay = seg.astype(np.float32) / 255
            rgb_f   = rgb.astype(np.float32)
            blended = rgb_f * (1 - seg_alpha) + (seg.astype(np.float32) / 255) * seg_alpha
            blended = np.clip(blended, 0, 1)
            st.image(array_to_pil(blended), caption="RGB + Segmentation Overlay",
                     use_column_width=True)
        elif rgb is not None:
            st.image(array_to_pil(rgb), caption="RGB only (no segmentation yet)",
                     use_column_width=True)
        else:
            st.info("Imagery not found. Run the data pipeline first.")

    with col_stats:
        st.markdown("**Land Cover Breakdown**")
        pit_df = load_pit_metrics()
        site_metrics = pit_df[pit_df["site_id"] == selected_site] if not pit_df.empty else pd.DataFrame()

        if not site_metrics.empty:
            labels = site_metrics["class"].str.replace("_", " ").str.title().tolist()
            values = site_metrics["fraction"].tolist()
            colors = CLASS_COLORS[:len(labels)]

            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                marker_colors=colors,
                hole=0.45,
                textinfo="label+percent",
            ))
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metric cards
            for _, row in site_metrics.iterrows():
                cls  = row["class"].replace("_", " ").title()
                pct  = f"{row['fraction']*100:.1f}%"
                area = f"{row['area_ha']:.0f} ha"
                st.markdown(
                    f'<div style="background:#1e3a5f;border-radius:8px;padding:8px 12px;'
                    f'margin-bottom:6px;color:white">'
                    f'<b>{cls}</b>: {pct} · {area}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Run the feature engineering pipeline (script 02) to see metrics.")

    # Run inference button
    st.divider()
    if st.button("🔄 Re-run Segmentation Inference", use_container_width=True):
        with st.spinner("Running U-Net inference..."):
            try:
                inf = load_inference_module()
                rgb_arr, seg_arr, metrics = inf.segment_site(selected_site)
                st.success("✅ Inference complete!")
                st.json(metrics)
            except Exception as e:
                st.error(f"Inference failed: {e}\nEnsure the training pipeline has been run.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 – Production Forecasting
# ══════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    st.subheader(f"Production Trend & Forecast – {selected_site.replace('_', ' ')}")

    prod_df = load_production_data()
    hist = prod_df[prod_df["site_id"] == selected_site].sort_values("year") if not prod_df.empty else pd.DataFrame()

    if hist.empty:
        st.info("Production data not found. Run the data pipeline first.")
    else:
        unit    = hist["unit"].iloc[0]
        mineral = hist["mineral"].iloc[0]

        # Forecast
        try:
            inf = load_inference_module()
            hist_df, fc_df = inf.forecast_production(selected_site, horizon=forecast_horizon)
        except Exception as e:
            fc_df = pd.DataFrame()
            st.warning(f"Forecast unavailable: {e}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["production"],
            mode="lines+markers", name="Historical",
            line=dict(color="#2196F3", width=2.5),
            marker=dict(size=6),
        ))

        if not fc_df.empty:
            if "LSTM" in fc_df.columns:
                fig.add_trace(go.Scatter(
                    x=fc_df["year"], y=fc_df["LSTM"],
                    mode="lines+markers", name="LSTM Forecast",
                    line=dict(color="#FF5722", width=2.5, dash="dash"),
                    marker=dict(size=8, symbol="square"),
                ))
            if "Prophet" in fc_df.columns:
                fig.add_trace(go.Scatter(
                    x=fc_df["year"], y=fc_df["Prophet"],
                    mode="lines+markers", name="Prophet Forecast",
                    line=dict(color="#4CAF50", width=2.5, dash="dash"),
                    marker=dict(size=8, symbol="triangle-up"),
                ))

        # Vertical line at last historical year
        fig.add_vline(x=hist["year"].max(), line_dash="dot", line_color="grey",
                      annotation_text="Forecast ▶", annotation_position="top right")

        fig.update_layout(
            title=f"{selected_site.replace('_', ' ')} – {mineral} Production",
            xaxis_title="Year", yaxis_title=f"Production ({unit})",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,30,50,0.8)",
            font=dict(color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        if not fc_df.empty:
            st.markdown("**Forecast Values**")
            st.dataframe(fc_df, use_container_width=True, hide_index=True)

        # Summary stats
        col_a, col_b, col_c = st.columns(3)
        peak = hist["production"].max()
        latest = hist["production"].iloc[-1]
        change = ((latest - hist["production"].iloc[0]) / hist["production"].iloc[0]) * 100
        with col_a:
            st.metric("Peak Production", f"{peak:,.0f} {unit}")
        with col_b:
            st.metric("Latest Production", f"{latest:,.0f} {unit}")
        with col_c:
            st.metric("Total Change (all years)", f"{change:+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 – Environment / NDVI
# ══════════════════════════════════════════════════════════════════════════════

with tab_env:
    st.subheader(f"Environmental Monitoring – {selected_site.replace('_', ' ')}")

    ndvi_df = load_ndvi_data()
    site_ndvi = ndvi_df[ndvi_df["site_id"] == selected_site] if not ndvi_df.empty else pd.DataFrame()

    if site_ndvi.empty:
        st.info("NDVI time-series not found. Run feature engineering (script 02).")
    else:
        # Trend line
        c = np.polyfit(site_ndvi["year"], site_ndvi["ndvi_mean"], 1)
        trend_per_yr = c[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NDVI Trend", f"{trend_per_yr:+.4f} / yr",
                      delta=f"{'Loss' if trend_per_yr < 0 else 'Gain'}",
                      delta_color="inverse")
        with col2:
            st.metric("Vegetation Cover (2024)",
                      f"{site_ndvi.iloc[-1]['veg_frac']*100:.1f}%")
        with col3:
            bare_chg = site_ndvi.iloc[-1]["bare_frac"] - site_ndvi.iloc[0]["bare_frac"]
            st.metric("Bare Soil Increase", f"{bare_chg*100:+.1f}%",
                      delta_color="inverse")

        # NDVI trend chart
        fig = go.Figure()
        fig.add_traces([
            go.Scatter(
                x=pd.concat([site_ndvi["year"], site_ndvi["year"][::-1]]),
                y=pd.concat([site_ndvi["ndvi_p90"], site_ndvi["ndvi_p10"][::-1]]),
                fill="toself", fillcolor="rgba(76,175,80,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="P10–P90 range", showlegend=True,
            ),
            go.Scatter(
                x=site_ndvi["year"], y=site_ndvi["ndvi_mean"],
                mode="lines+markers", name="Mean NDVI",
                line=dict(color="#4CAF50", width=2.5), marker=dict(size=6),
            ),
            go.Scatter(
                x=site_ndvi["year"],
                y=np.polyval(c, site_ndvi["year"]),
                mode="lines", name="Linear Trend",
                line=dict(color="red", dash="dash", width=1.5),
            ),
        ])
        fig.update_layout(
            title="NDVI Mean ± Range (vegetation health indicator)",
            xaxis_title="Year", yaxis_title="NDVI",
            yaxis=dict(range=[-0.15, 0.65]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,30,50,0.8)",
            font=dict(color="white"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Vegetation / bare soil area chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=site_ndvi["year"], y=site_ndvi["veg_frac"] * 100,
            name="Vegetation %", marker_color="#4CAF50",
        ))
        fig2.add_trace(go.Bar(
            x=site_ndvi["year"], y=site_ndvi["bare_frac"] * 100,
            name="Bare Soil %", marker_color="#FF7043",
        ))
        fig2.update_layout(
            barmode="group",
            title="Land Cover Change Over Time",
            xaxis_title="Year", yaxis_title="Cover (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,30,50,0.8)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Erosion risk matrix across all sites
    st.markdown("---")
    st.subheader("Environmental Risk Overview – All Sites")
    if not ndvi_df.empty:
        risk_rows = []
        for sid in SA_SITES:
            sdf = ndvi_df[ndvi_df["site_id"] == sid]
            if sdf.empty:
                continue
            sdf = sdf.sort_values("year")
            c = np.polyfit(sdf["year"], sdf["ndvi_mean"], 1)
            trend = c[0]
            bare_chg = sdf.iloc[-1]["bare_frac"] - sdf.iloc[0]["bare_frac"]
            risk_score = abs(trend) * 500 + bare_chg * 100
            risk_rows.append({
                "Site": sid.replace("_", " "),
                "Mineral": SA_SITES[sid]["mineral"],
                "NDVI Trend /yr": f"{trend:+.4f}",
                "Bare Soil Δ": f"{bare_chg*100:+.1f}%",
                "Risk Score": f"{risk_score:.2f}",
                "Risk Level": "🔴 High" if risk_score > 1.5 else ("🟡 Medium" if risk_score > 0.8 else "🟢 Low"),
            })
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 – Custom Upload
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.subheader("Upload Custom Imagery for Analysis")
    st.markdown("""
    Upload a satellite image for segmentation analysis.
    Supports: **PNG, JPG** (RGB or RGBA), or **GeoTIFF** (up to 4 bands).
    The system will run the U-Net segmentation model and display results.
    """)

    uploaded = st.file_uploader(
        "Choose image file", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded is not None:
        file_ext = uploaded.name.split(".")[-1].lower()

        if file_ext in ("tif", "tiff"):
            # GeoTIFF processing
            with rasterio.open(uploaded) as src:
                bands = src.count
                if bands >= 4:
                    img_arr = src.read([1,2,3,4]).astype(np.float32)
                    rgb = np.stack([img_arr[0], img_arr[1], img_arr[2]], -1)
                    rgb = np.clip(rgb / (rgb.max() + 1e-6), 0, 1)
                else:
                    img_arr = src.read().astype(np.float32)
                    # Pad to 4 bands if needed
                    while img_arr.shape[0] < 4:
                        img_arr = np.concatenate([img_arr, img_arr[[0]]], axis=0)
                    rgb = np.stack([img_arr[0], img_arr[1], img_arr[2]], -1)
                    rgb = np.clip(rgb / (rgb.max() + 1e-6), 0, 1)
        else:
            # PNG / JPG
            pil_img = Image.open(uploaded).convert("RGB")
            rgb_raw = np.array(pil_img).astype(np.float32) / 255
            rgb = rgb_raw
            # Simulate 4-band from RGB by estimating NIR (channel avg)
            nir_proxy = (rgb_raw[..., 0] * 0.3 + rgb_raw[..., 1] * 0.6 + rgb_raw[..., 2] * 0.1)
            img_arr = np.stack([rgb_raw[..., 0], rgb_raw[..., 1], rgb_raw[..., 2], nir_proxy], axis=0)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Uploaded Image**")
            st.image(array_to_pil(rgb), use_column_width=True)

        with col2:
            st.markdown("**Segmentation Result**")
            if st.button("▶ Run Segmentation", use_container_width=True):
                with st.spinner("Running U-Net..."):
                    try:
                        inf = load_inference_module()
                        # Resize to 256×256 for model compatibility
                        from PIL import Image as PILImage
                        pil_raw = PILImage.fromarray((np.clip(img_arr.transpose(1,2,0), 0, 1)*255).astype(np.uint8)[:,:,:3])
                        pil_resized = pil_raw.resize((256, 256))
                        arr_resized = np.array(pil_resized).astype(np.float32) / 255
                        # Rebuild 4-band
                        nir_r = arr_resized[...,0]*0.3 + arr_resized[...,1]*0.6 + arr_resized[...,2]*0.1
                        arr4 = np.stack([arr_resized[...,0], arr_resized[...,1], arr_resized[...,2], nir_r], 0)

                        rgb_out, seg_arr, metrics = inf.segment_site("_uploaded_", image_array=arr4)
                        overlay = inf.make_overlay(rgb_out, seg_arr, alpha=seg_alpha)
                        st.image(array_to_pil(overlay), use_column_width=True)
                        st.success("✅ Segmentation complete")
                        st.markdown("**Land Cover Metrics:**")
                        st.json(metrics)
                    except Exception as e:
                        st.error(f"Segmentation error: {e}")
                        st.info("Make sure the training pipeline (script 03) has been run to create models.")
            else:
                st.info("Click 'Run Segmentation' to analyze.")

    else:
        st.info("Upload a satellite image above to get started.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;opacity:0.5;font-size:0.8rem'>"
    "SA Mining Analyzer · Local-only · No cloud required · "
    "Data: Synthetic Sentinel-2 proxies + USGS MRDS · Models: U-Net + LSTM + Prophet"
    "</div>",
    unsafe_allow_html=True
)