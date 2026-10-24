import io
from pathlib import Path

import streamlit as st
import geopandas as gpd
import folium
import branca.colormap as cm
import pandas as pd
from streamlit_folium import st_folium
from logic import score, recommend

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(layout="wide", page_title="Urban Heat Risk Mapper", page_icon="🌡️")


def apply_theme(dark_mode: bool):
    palette = {
        "bg": "#0f172a" if dark_mode else "#f6f9ff",
        "bg_2": "#111827" if dark_mode else "#edf3ff",
        "panel": "rgba(15, 23, 42, 0.78)" if dark_mode else "rgba(255,255,255,0.9)",
        "panel_border": "#334155" if dark_mode else "#dfe7f5",
        "text": "#e2e8f0" if dark_mode else "#102a43",
        "muted": "#b7c4d9" if dark_mode else "#5a6c8c",
        "card": "rgba(30,41,59,0.88)" if dark_mode else "rgba(255,255,255,0.96)",
        "shadow": "rgba(15, 23, 42, 0.28)" if dark_mode else "rgba(31, 41, 55, 0.05)",
    }
    bg_gradient = (
        f"linear-gradient(180deg, {palette['bg']} 0%, {palette['bg_2']} 100%)"
    )
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {bg_gradient};
                color: {palette['text']};
            }}
            .metric-card, .recommendation, .summary-panel, .download-box {{
                background: {palette['card']};
                border: 1px solid {palette['panel_border']};
                border-radius: 18px;
                padding: 1.15rem 1.2rem;
                box-shadow: 0 10px 24px {palette['shadow']};
                margin-bottom: 0.8rem;
            }}
            .metric-label, .section-kicker, .hero-label {{
                color: {palette['muted']};
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.45rem;
                font-weight: 700;
            }}
            .metric-value {{
                font-size: 2.2rem;
                font-weight: 800;
                color: {palette['text']};
                line-height: 1.1;
            }}
            .metric-delta {{
                color: {palette['muted']};
                font-size: 0.9rem;
                margin-top: 0.3rem;
                line-height: 1.5;
            }}
            .recommendation {{
                padding: 0.9rem 1rem;
            }}
            .rec-badge {{
                display: inline-block;
                border-radius: 999px;
                padding: 0.28rem 0.7rem;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 0.5rem;
            }}
            .badge-tree {{ background: rgba(16, 185, 129, 0.12); color: #6ee7b7; }}
            .badge-shelter {{ background: rgba(59, 130, 246, 0.12); color: #93c5fd; }}
            .badge-survey {{ background: rgba(148, 163, 184, 0.16); color: #cbd5e1; }}
            .rec-title {{
                font-size: 1.15rem;
                font-weight: 700;
                color: {palette['text']};
                margin-bottom: 0.1rem;
            }}
            .rec-meta {{
                color: {palette['muted']};
                font-size: 0.85rem;
                margin-bottom: 0.4rem;
            }}
            .rec-body {{
                color: {palette['text']};
                font-size: 0.98rem;
                line-height: 1.6;
                margin: 0;
            }}
            .hero {{
                margin-bottom: 1rem;
            }}
            .hero h1 {{
                margin: 0;
                font-size: clamp(2.2rem, 4vw, 3.2rem);
                color: {palette['text']};
                line-height: 1.15;
            }}
            .hero p {{
                margin-top: 0.6rem;
                color: {palette['muted']};
                font-size: 1.1rem;
                line-height: 1.6;
            }}
            .summary-panel {{
                padding: 1.1rem 1.25rem;
                color: {palette['text']};
                font-size: 1.02rem;
                line-height: 1.6;
            }}
            .summary-panel strong {{
                color: #f59e0b;
            }}
            .legend-chip {{
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                padding: 0.5rem 0.8rem;
                border-radius: 999px;
                background: {palette['card']};
                border: 1px solid {palette['panel_border']};
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
                color: {palette['text']};
                font-size: 0.88rem;
            }}
            .stApp, .stMarkdown, .stDataFrame, .stSelectbox, .stRadio, .stSlider, .stButton, .stDownloadButton {{
                font-size: 1rem;
                line-height: 1.5;
            }}
            .swatch {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
            }}
            [data-testid="stSidebar"] {{
                background: {palette['card']};
                border-right: 1px solid {palette['panel_border']};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# set default theme before UI rendering
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

apply_theme(st.session_state.dark_mode)


@st.cache_data
def load():
    files = {
        "cells": BASE_DIR / "cells.geojson",
        "parks": BASE_DIR / "parks.geojson",
        "buildings": BASE_DIR / "buildings.geojson",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required GeoJSON files: " + ", ".join(missing)
        )
    return (
        gpd.read_file(files["cells"]),
        gpd.read_file(files["parks"]),
        gpd.read_file(files["buildings"]),
    )


cells, parks, blds = load()

# ---- Sidebar controls ----
st.sidebar.title("Urban Heat Risk")
st.sidebar.caption("Targeted cooling and greening actions")
st.session_state.dark_mode = st.sidebar.toggle("Dark mode", value=st.session_state.dark_mode)
apply_theme(st.session_state.dark_mode)
city = st.sidebar.selectbox("City", ["Demo City"])
layer = st.sidebar.radio("Map layer", ["Priority", "Heat index"])
top_n = st.sidebar.slider("Number of priority sites", 5, 30, 10)
st.sidebar.subheader("Heat index weights")
w1 = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
w2 = st.sidebar.slider("Built-up area", 0.0, 1.0, 0.25)
w3 = st.sidebar.slider("Vegetation (reduces heat)", 0.0, 1.0, 0.25)
show_trees = st.sidebar.checkbox("Tree sites", True)
show_shelters = st.sidebar.checkbox("Cooling shelters", True)

st.sidebar.markdown("---")

# ---- Compute ----
scored = score(cells, w1, w2, w3)
metric_col = "priority" if layer == "Priority" else "heat_index"
top = recommend(scored.nsmallest(top_n, "rank"), parks, blds)

hottest_cell = scored.loc[scored[metric_col].idxmax()]
max_priority = scored["priority"].max()
priority_90 = scored["priority"].quantile(0.9)
priority_pop = int(scored.loc[scored["priority"].nlargest(10).index, "pop"].sum())
action_counts = top["action"].value_counts()
primary_action = action_counts.idxmax() if not action_counts.empty else "Needs site survey"

# ---- Header ----
hero = st.container()
with hero:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Urban resilience dashboard</div>
            <h1>Urban Heat Risk Mapper</h1>
            <p>Prioritizing cooling interventions where extreme heat and vulnerability overlap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

summary_text = (
    f"This plan prioritizes <strong>{(scored['priority'] >= priority_90).sum():,}</strong> high-risk cells, "
    f"covering about <strong>{priority_pop:,}</strong> residents. "
    f"The most common intervention is <strong>{primary_action}</strong> for the next round of action."
)
st.markdown(
    f"""
    <div class="summary-panel">
        <div class="section-kicker">Operational summary</div>
        <div>{summary_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Summary metrics ----
metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Peak heat score</div>
            <div class="metric-value">{hottest_cell[metric_col]:.2f}</div>
            <div class="metric-delta">Highest scoring cell in the current layer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with metric_cols[1]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Priority hotspots</div>
            <div class="metric-value">{(scored['priority'] >= priority_90).sum():,}</div>
            <div class="metric-delta">Cells in the top 10% of priority risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with metric_cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">At-risk population</div>
            <div class="metric-value">{priority_pop:,}</div>
            <div class="metric-delta">Estimated residents in the highest-priority sites</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with metric_cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Lead action</div>
            <div class="metric-value">{primary_action}</div>
            <div class="metric-delta">Most common intervention in the plan</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Map + recommendations ----
left, right = st.columns([3, 1.2])

with left:
    st.markdown('<div class="section-kicker">Hotspot map</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="legend-chip"><span class="swatch" style="background:#ffffb2"></span>Low</div>
        <div class="legend-chip"><span class="swatch" style="background:#fd8d3c"></span>Moderate</div>
        <div class="legend-chip"><span class="swatch" style="background:#bd0026"></span>Critical</div>
        """,
        unsafe_allow_html=True,
    )
    minx, miny, maxx, maxy = scored.total_bounds
    m = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2],
        zoom_start=12,
        tiles="CartoDB positron",
    )
    cmap = cm.LinearColormap(
        ["#ffffb2", "#fd8d3c", "#bd0026"],
        vmin=0,
        vmax=1,
        caption=layer,
    )
    folium.GeoJson(
        scored[["id", metric_col, "geometry"]],
        style_function=lambda f: {
            "fillColor": cmap(f["properties"][metric_col]),
            "color": "#1f2937",
            "weight": 0.2,
            "fillOpacity": 0.72,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["id", metric_col],
            aliases=["Cell ID", layer],
            localize=True,
        ),
    ).add_to(m)
    cmap.add_to(m)

    for r in top.itertuples():
        is_tree = r.action == "Plant trees"
        is_shelter = r.action == "Open cooling shelter"
        if (is_tree and not show_trees) or (is_shelter and not show_shelters):
            continue
        color = "green" if is_tree else "blue" if is_shelter else "gray"
        folium.Marker(
            [r.lat, r.lon],
            popup=f"<b>{r.action}</b><br>{r.why}",
            icon=folium.Icon(color=color),
        ).add_to(m)

    st_folium(m, height=620, use_container_width=True)

with right:
    st.markdown('<div class="section-kicker">Recommended sites</div>', unsafe_allow_html=True)
    if not top.empty:
        export_df = top[["rank", "action", "priority", "heat_index", "why"]].copy()
        export_df = export_df.rename(columns={"rank": "site_rank", "priority": "priority_score", "heat_index": "heat_score", "why": "recommendation_reason"})
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export recommendations (.csv)",
            data=csv_bytes,
            file_name="urban_heat_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No recommended sites available with the current settings.")

    if top.empty:
        st.info("No recommended sites available with the current settings.")
    for r in top.itertuples():
        action = r.action
        badge_class = "badge-tree" if action == "Plant trees" else "badge-shelter" if action == "Open cooling shelter" else "badge-survey"
        st.markdown(
            f"""
            <div class="recommendation">
                <div class="rec-badge {badge_class}">{action}</div>
                <div class="rec-title">Site #{int(r.rank)}</div>
                <div class="rec-meta">Priority {r.priority:.2f} • Heat {r.heat_index:.2f}</div>
                <p class="rec-body">{r.why}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- Intervention mix ----
st.markdown('<div class="section-kicker">Intervention mix</div>', unsafe_allow_html=True)
if not top.empty:
    action_cols = st.columns(3)
    action_labels = ["Plant trees", "Open cooling shelter", "Needs site survey"]
    for idx, label in enumerate(action_labels):
        count = int((top["action"] == label).sum())
        with action_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{count}</div>
                    <div class="metric-delta">Priority sites</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.expander("Methodology and limitations"):
    st.write(
        "Heat index = w1 × temperature + w2 × built-up − w3 × vegetation (each scaled 0-1). "
        "Priority = heat index × population. Data: Landsat 9, Sentinel-2, ESA WorldCover, "
        "WorldPop, and OpenStreetMap. Limitations: this is surface temperature, not air temperature; "
        "thermal data is 30 m; population is modeled, not a census count."
    )