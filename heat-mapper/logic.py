import pandas as pd


def rescale(s):
    """Rescale numbers to 0-1. If all values are equal, return zeros."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0
    return (s - lo) / (hi - lo)


def norm(s):
    """Rescale to 0-1, ignoring the most extreme outliers (2nd-98th percentile)."""
    lo, hi = s.quantile(0.02), s.quantile(0.98)
    if hi == lo:
        return s * 0
    return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0)


def add_air_temp_estimate(cells, metrics, min_stations=10, min_corr=0.5):
    """
    Add a 'tair_est' column (estimated air temperature, degrees C) using the
    straight-line fit saved by validate.py (air = intercept + slope * lst).

    It is only added when the station check is trustworthy:
      - at least ⁠ min_stations ⁠ stations were used, and
      - the correlation between satellite and station temperature is >= ⁠ min_corr ⁠.
    Otherwise 'tair_est' stays empty (NaN) and the app works exactly as before.

    Returns: (cells_with_column, message_for_the_user)

    IMPORTANT: this is a straight-line transform of surface temperature, so it
    does NOT change which cells rank highest. It only converts the number into
    a more familiar unit for display. It is a rough estimate, not a measurement.
    """
    c = cells.copy()
    c["tair_est"] = float("nan")

    if not metrics:
        return c, "No station data loaded, so no air-temperature estimate."

    n, r = metrics.get("n"), metrics.get("pearson")
    a, b = metrics.get("intercept"), metrics.get("slope")
    if n is None or r is None or a is None or b is None:
        return c, "Station fit incomplete (need at least 3 stations), so no air-temperature estimate."
    if n < min_stations:
        return c, f"Only {n} stations (need {min_stations}+), so no air-temperature estimate."
    if r < min_corr:
        return c, f"Weak agreement with stations (r = {r:.2f}), so no air-temperature estimate."
    if b <= 0:
        return c, "Station fit slopes the wrong way (hotter surface, cooler air), so no air-temperature estimate."

    c["tair_est"] = a + b * c["lst"]   # keep full precision; round only when displaying
    return c, f"Air temperature estimated from {n} stations (r = {r:.2f}). Rough estimate only."


def score(cells, w1, w2, w3, temp_col="lst"):
    """
    Heat index = w1*temperature + w2*built-up - w3*vegetation  (each scaled 0-1)
    Priority   = heat index x population
    temp_col   = which temperature column to use ("lst" by default).
    """
    c = cells.copy()
    hi = w1 * norm(c[temp_col]) + w2 * norm(c.builtup) - w3 * norm(c.ndvi)
    c["heat_index"] = rescale(hi)
    c["priority"] = rescale(c["heat_index"] * norm(c["pop"]))
    c["rank"] = c["priority"].rank(ascending=False, method="first")
    return c


def recommend(top, parks, blds):
    """For each top cell: plant trees, open a shelter, or needs a site survey."""
    utm = top.estimate_utm_crs()
    top_m = top.to_crs(utm)
    parks_m, blds_m = parks.to_crs(utm), blds.to_crs(utm)
    cent = top_m.geometry.centroid.to_crs(4326)
    actions, whys = [], []
    for _, row in top_m.iterrows():
        base = f"Surface temp {row.lst:.1f}C"
        tair = row.get("tair_est", float("nan"))
        if pd.notna(tair):
            base += f" (est. air temp ~{tair:.1f}C)"
        base += f", greenness {row.ndvi:.2f}, ~{row['pop']:,.0f} people. "

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