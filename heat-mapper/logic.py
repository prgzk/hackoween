import geopandas as gpd

def norm(s):
    """Rescale to 0-1, ignoring extreme outliers."""
    lo, hi = s.quantile(0.02), s.quantile(0.98)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0)

def score(cells, w1, w2, w3):
    c = cells.copy()
    # Heat index = hotter + more built-up - more greenery
    hi = w1 * norm(c.lst) + w2 * norm(c.builtup) - w3 * norm(c.ndvi)
    c["heat_index"] = (hi - hi.min()) / (hi.max() - hi.min())
    # Priority = heat x people (all cells are equal size, so pop = density)
    p = c.heat_index * norm(c["pop"])
    c["priority"] = (p - p.min()) / (p.max() - p.min())
    c["rank"] = c.priority.rank(ascending=False)
    return c

def recommend(top, parks, blds):
    """For each top cell: trees, shelter, or needs survey."""
    utm = top.estimate_utm_crs()
    top_m = top.to_crs(utm)
    parks_m, blds_m = parks.to_crs(utm), blds.to_crs(utm)
    cent = top_m.geometry.centroid.to_crs(4326)
    actions, whys = [], []
    for i, row in top_m.iterrows():
        base = f"LST {row.lst:.1f}°C, greenness {row.ndvi:.2f}, ~{row['pop']:,.0f} people. "
        if parks_m.intersects(row.geometry).any() and row.ndvi < 0.3:
            actions.append("Plant trees")
            whys.append(base + "Open space is available here but canopy is low.")
        elif (blds_m.distance(row.geometry) < 500).any():
            actions.append("Open cooling shelter")
            whys.append(base + "No open space, but a public building is within 500 m.")
        else:
            actions.append("Needs site survey")
            whys.append(base + "No open space or public building found nearby.")
    out = top.copy()
    out["action"], out["why"] = actions, whys
    out["lat"], out["lon"] = cent.y.values, cent.x.values
    return out