import json
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

STATIONS_CSV = "stations.csv"
CELLS = "cells.geojson"

# If your CSV uses different column names, edit the right-hand side only.
# Example: "tair": "max_temp_c"
COLS = {"name": "name", "lat": "lat", "lon": "lon", "tair": "tair"}


def clean(x):
    """Turn numpy numbers / NaN into something json can store."""
    x = float(x)
    return None if np.isnan(x) else round(x, 3)


def main():
    try:
        cells = gpd.read_file(CELLS)
    except Exception as e:
        sys.exit(f"Could not read {CELLS}: {e}\nRun make_fake.py or fetch_data.py first.")

    try:
        raw = pd.read_csv(STATIONS_CSV)
    except Exception as e:
        sys.exit(f"Could not read {STATIONS_CSV}: {e}")

    st = raw.rename(columns={v: k for k, v in COLS.items()})
    missing = {"name", "lat", "lon", "tair"} - set(st.columns)
    if missing:
        sys.exit(f"{STATIONS_CSV} is missing columns: {sorted(missing)}\n"
                 f"Columns found: {list(raw.columns)}\n"
                 f"Edit the COLS settings at the top of validate.py.")

    st = st[["name", "lat", "lon", "tair"]].copy()
    for c in ["lat", "lon", "tair"]:
        st[c] = pd.to_numeric(st[c], errors="coerce")
    st = st.dropna()
    print(f"Stations read: {len(st)}")

    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]),
                           crs=4326)
    cols = cells[["id", "lst", "ndvi", "builtup", "geometry"]].rename(columns={"id": "cell_id"})
    joined = gpd.sjoin(pts, cols, how="inner", predicate="within")
    joined = joined.drop(columns=["index_right"]).dropna(subset=["lst", "tair"])
    print(f"Stations inside your study area: {len(joined)}")

    if len(joined) == 0:
        sys.exit("No stations fall inside the grid. Check lat/lon (lat first) and the study area.")

    n = len(joined)
    lst, tair = joined["lst"].to_numpy(), joined["tair"].to_numpy()

    metrics = {"n": n,
               "pearson": None, "spearman": None,
               "slope": None, "intercept": None,
               "bias": clean(np.mean(lst - tair)),     # surface minus air
               "tair_mean": clean(np.mean(tair))}

    if n >= 3:
        metrics["pearson"] = clean(np.corrcoef(lst, tair)[0, 1])
        metrics["spearman"] = clean(joined["lst"].corr(joined["tair"], method="spearman"))
        slope, intercept = np.polyfit(lst, tair, 1)
        metrics["slope"], metrics["intercept"] = clean(slope), clean(intercept)
    else:
        print("WARNING: fewer than 3 stations - correlation is not meaningful.")

    joined["diff"] = (joined["lst"] - joined["tair"]).round(2)
    joined[["name", "tair", "lst", "diff", "cell_id", "geometry"]].to_file(
        "stations_matched.geojson", driver="GeoJSON")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nHow to read this:")
    print("  bias     = average (satellite surface temp - station air temp). Usually positive")
    print("             in daytime because hot ground is warmer than the air above it.")
    print("  spearman = do hotter stations also get hotter satellite values? (1 = same order)")
    if n < 10:
        print("  NOTE: under 10 stations, treat these numbers as a rough sanity check only.")


if __name__ == "_main_":
    main()