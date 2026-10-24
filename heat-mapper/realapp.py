import streamlit as st, geopandas as gpd, folium, branca.colormap as cm
from streamlit_folium import st_folium
from logic import score, recommend

st.set_page_config(layout="wide", page_title="Urban Heat Island Mapper")

@st.cache_data
def load():
    return (gpd.read_file("cells.geojson"),
            gpd.read_file("parks.geojson"),
            gpd.read_file("buildings.geojson"))

cells, parks, blds = load()

# ---- Sidebar controls ----
st.sidebar.title("Urban Heat Island Mapper")
st.sidebar.selectbox("City", ["Your city"])       # one city is fine for the demo
layer = st.sidebar.radio("Map layer", ["Priority", "Heat index"])
top_n = st.sidebar.slider("Number of priority sites", 5, 30, 10)
st.sidebar.subheader("Heat index weights")
w1 = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
w2 = st.sidebar.slider("Built-up area", 0.0, 1.0, 0.25)
w3 = st.sidebar.slider("Vegetation (reduces heat)", 0.0, 1.0, 0.25)
show_trees = st.sidebar.checkbox("Tree sites", True)
show_shelters = st.sidebar.checkbox("Cooling shelters", True)

# ---- Compute ----
scored = score(cells, w1, w2, w3)
col = "priority" if layer == "Priority" else "heat_index"
top = recommend(scored.nsmallest(top_n, "rank"), parks, blds)

# ---- Map ----
minx, miny, maxx, maxy = scored.total_bounds
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=12,
               tiles="CartoDB positron")
cmap = cm.LinearColormap(["#ffffb2", "#fd8d3c", "#bd0026"], vmin=0, vmax=1, caption=layer)
folium.GeoJson(
    scored[["id", col, "geometry"]],
    style_function=lambda f: {"fillColor": cmap(f["properties"][col]),
                              "color": "none", "fillOpacity": 0.6},
    tooltip=folium.GeoJsonTooltip(fields=[col]),
).add_to(m)
cmap.add_to(m)

for r in top.itertuples():
    is_tree = r.action == "Plant trees"
    is_shelter = r.action == "Open cooling shelter"
    if (is_tree and not show_trees) or (is_shelter and not show_shelters):
        continue
    color = "green" if is_tree else "blue" if is_shelter else "gray"
    folium.Marker([r.lat, r.lon], popup=f"<b>{r.action}</b><br>{r.why}",
                  icon=folium.Icon(color=color)).add_to(m)

left, right = st.columns([3, 1])
with left:
    st_folium(m, height=600, use_container_width=True)
with right:
    st.subheader("Top recommended sites")
    for r in top.itertuples():
        st.markdown(f"**Site {int(r.rank)} | score {r.priority:.2f}**  \n{r.action}")
        st.caption(r.why)

with st.expander("Methodology and limitations"):
    st.write("Heat index = w1 × temperature + w2 × built-up − w3 × vegetation (each scaled 0-1). "
             "Priority = heat index × population. Data: Landsat 9, Sentinel-2, ESA WorldCover, "
             "WorldPop, OpenStreetMap. Limitations: this is surface temperature, not air temperature; "
             "thermal data is 30 m; population is modelled, not a census count.")