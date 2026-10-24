import ee, geemap, geopandas as gpd, osmnx as ox
from shapely.geometry import Point, box

# ---------- SETTINGS: change these ----------
PROJECT = "trial-heatmap"
LAT, LON = 12.9716, 77.5946        # city centre (example: Bengaluru)
RADIUS_M = 6000                    # 6 km each way -> 12 km x 12 km area
CELL_M = 250                       # cell size in metres
START, END = "2026-03-01", "2026-05-31"   # a hot, dry season
# --------------------------------------------

ee.Initialize(project=PROJECT)

# 1) Build the grid of 250 m squares (in metres, then convert to lat/lon)
center = gpd.GeoSeries([Point(LON, LAT)], crs=4326)
utm = center.estimate_utm_crs()
cx, cy = center.to_crs(utm).iloc[0].coords[0]
cells = []
x = cx - RADIUS_M
while x < cx + RADIUS_M:
    y = cy - RADIUS_M
    while y < cy + RADIUS_M:
        cells.append(box(x, y, x + CELL_M, y + CELL_M))
        y += CELL_M
    x += CELL_M
grid = gpd.GeoDataFrame({"id": range(len(cells))}, geometry=cells, crs=utm).to_crs(4326)
aoi = ee.Geometry.Rectangle(list(grid.total_bounds))

# 2) Satellite layers (median of many images removes clouds)
lst = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
       .filterBounds(aoi).filterDate(START, END)
       .filter(ee.Filter.lt("CLOUD_COVER", 20))
       .select("ST_B10").median()
       .multiply(0.00341802).add(149.0).subtract(273.15)   # raw number -> °C
       .rename("lst"))

ndvi = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi).filterDate(START, END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median().normalizedDifference(["B8", "B4"]).rename("ndvi"))

builtup = (ee.ImageCollection("ESA/WorldCover/v200").first()
           .select("Map").eq(50).rename("builtup"))        # class 50 = built-up

pop = (ee.ImageCollection("WorldPop/GP/100m/pop")
       .filterBounds(aoi).filter(ee.Filter.eq("year", 2020))
       .mosaic().select("population").rename("pop"))

# 3) Average (or sum, for population) each layer inside each cell
fc = geemap.gdf_to_ee(grid)
means = lst.addBands(ndvi).addBands(builtup).reduceRegions(
    collection=fc, reducer=ee.Reducer.mean(), scale=30)
pops = pop.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100)

a = geemap.ee_to_gdf(means)
b = geemap.ee_to_gdf(pops)[["id", "sum"]].rename(columns={"sum": "pop"})
out = a[["id", "lst", "ndvi", "builtup", "geometry"]].merge(b, on="id")
out.to_file("cells.geojson", driver="GeoJSON")
print("Saved", len(out), "cells")

# 4) Parks and public buildings from OpenStreetMap
parks = ox.features_from_point((LAT, LON),
    {"leisure": ["park", "playground", "pitch"], "landuse": ["grass", "recreation_ground"]},
    dist=RADIUS_M + 500)
blds = ox.features_from_point((LAT, LON),
    {"amenity": ["community_centre", "school", "library", "place_of_worship", "townhall"]},
    dist=RADIUS_M + 500)
parks[["geometry"]].reset_index(drop=True).to_file("parks.geojson", driver="GeoJSON")
blds[["geometry"]].reset_index(drop=True).to_file("buildings.geojson", driver="GeoJSON")