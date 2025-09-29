# src/sch/gtfs_graph.py
from __future__ import annotations

from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

CRS_WGS84 = "EPSG:4326"
CRS_CH = "EPSG:2056"

# Globals populated by build_station_graph
adj: Dict[str, list] = {}
stations_pts: gpd.GeoDataFrame | None = None
stops: pd.DataFrame | None = None
st_times: pd.DataFrame | None = None


def _ensure_station_id(stops_df: pd.DataFrame) -> pd.DataFrame:
    s = stops_df.copy()
    if "station_id" not in s.columns:
        parent = s.get("parent_station")
        if parent is None:
            parent = pd.Series([None] * len(s), index=s.index)
        s["station_id"] = parent.fillna(s["stop_id"]).astype(str)
    return s


def _hms_to_sec(x: str) -> int:
    h, m, s = str(x).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_station_graph(
    stops_in: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame, stop_times_in: pd.DataFrame
) -> dict:
    """
    Returns dict with keys: adj, stations_pts, stops, st_times
    - adj: adjacency {u: [(v, w_sec), ...]}
    - stations_pts: GeoDataFrame of unique station points (LV95)
    - stops: DataFrame with ensured 'station_id' and 'stop_name'
    - st_times: filtered stop_times used for edges
    """
    global adj, stations_pts, stops, st_times

    stops = _ensure_station_id(stops_in)
    stop_times = stop_times_in.copy()

    # keep all route types (0..7) like in the notebook
    use_types = set([0, 1, 2, 3, 4, 5, 6, 7])
    sel_routes = routes[routes["route_type"].isin(use_types)]["route_id"].unique()
    sel_trips = trips[trips["route_id"].isin(sel_routes)]["trip_id"].unique()

    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    stop_times = stop_times[stop_times["trip_id"].isin(sel_trips)].copy()

    st = stop_times.dropna(subset=["arrival_time", "departure_time"]).copy()
    st["station_id"] = st["stop_id"].map(stops.set_index("stop_id")["station_id"].to_dict())
    st = st.dropna(subset=["station_id"]).copy()

    st["arr_s"] = st["arrival_time"].map(_hms_to_sec)
    st["dep_s"] = st["departure_time"].map(_hms_to_sec)
    st.sort_values(["trip_id", "stop_sequence"], inplace=True)
    st["next_station"] = st.groupby("trip_id")["station_id"].shift(-1)
    st["next_arr_s"] = st.groupby("trip_id")["arr_s"].shift(-1)

    edges_raw = st.dropna(subset=["next_station", "next_arr_s"])[
        ["station_id", "next_station", "dep_s", "next_arr_s"]
    ].copy()
    edges_raw["w_sec"] = (edges_raw["next_arr_s"] - edges_raw["dep_s"]).astype(float)

    # station centroids
    df_pts = stops[["station_id", "stop_lon", "stop_lat"]].dropna().drop_duplicates().copy()
    df_pts["stop_lon"] = pd.to_numeric(df_pts["stop_lon"], errors="coerce")
    df_pts["stop_lat"] = pd.to_numeric(df_pts["stop_lat"], errors="coerce")
    df_pts = df_pts.dropna(subset=["stop_lon", "stop_lat"]).reset_index(drop=True)

    g_wgs = gpd.GeoDataFrame(
        df_pts, geometry=gpd.points_from_xy(df_pts["stop_lon"], df_pts["stop_lat"]), crs=CRS_WGS84
    )
    stations_pts = g_wgs.to_crs(CRS_CH).dissolve(by="station_id", as_index=False)
    stations_pts["geometry"] = stations_pts.geometry.centroid
    stxy = stations_pts.set_index("station_id")["geometry"].apply(lambda p: (p.x, p.y)).to_dict()

    # sanitize edges (min time + max speed)
    def _sanitize(edges_df, stxy_dict, min_sec=20.0, vmax_kmh=160.0):
        u_xy = edges_df["station_id"].map(stxy_dict)
        v_xy = edges_df["next_station"].map(stxy_dict)
        dx = np.array([a[0] - b[0] if (a and b) else np.nan for a, b in zip(u_xy, v_xy)])
        dy = np.array([a[1] - b[1] if (a and b) else np.nan for a, b in zip(u_xy, v_xy)])
        dist_m = np.hypot(dx, dy)
        w = edges_df["w_sec"].astype(float).to_numpy()
        w = np.where(~np.isfinite(w) | (w < min_sec), min_sec, w)
        vmax_mps = vmax_kmh / 3.6
        bad = (dist_m > 0) & (dist_m / w > vmax_mps)
        w[bad] = np.maximum(dist_m[bad] / vmax_mps, min_sec)
        out = edges_df.copy()
        out["w_sec"] = w
        return out.dropna(subset=["w_sec"])

    edge_sane = _sanitize(edges_raw, stxy)

    # adjacency
    from collections import defaultdict

    adj_local = defaultdict(list)
    for r in edge_sane.itertuples(index=False):
        u = str(r.station_id)
        v = str(r.next_station)
        w = float(r.w_sec)
        adj_local[u].append((v, w))
    adj = dict(adj_local)

    # keep handy
    st_times = st.copy()

    return dict(adj=adj, stations_pts=stations_pts, stops=stops, st_times=st_times)


# ---- commute function (same semantics as notebook) ----
# It uses the globals above (adj, stations_pts, stops, st_times)
from shapely.ops import unary_union as _uu


def _normtxt(s: str) -> str:
    s = str(s).lower()
    for a, b in (("ä", "a"), ("ö", "o"), ("ü", "u"), ("é", "e"), ("è", "e"), ("ê", "e")):
        s = s.replace(a, b)
    return s.strip()


def match_station_ids_by_name(name: str) -> list[str]:
    name_map = stops.groupby("station_id")["stop_name"].agg(lambda s: s.value_counts().idxmax())
    n0 = _normtxt(name)
    exact = [sid for sid, nm in name_map.items() if isinstance(nm, str) and _normtxt(nm) == n0]
    if exact:
        return sorted({str(x) for x in exact})
    starts = [
        sid for sid, nm in name_map.items() if isinstance(nm, str) and _normtxt(nm).startswith(n0)
    ]
    if starts:
        return sorted({str(x) for x in starts})
    contains = [sid for sid, nm in name_map.items() if isinstance(nm, str) and n0 in _normtxt(nm)]
    return sorted({str(x) for x in contains})


def _multi_source_dijkstra(src_ids):
    import heapq

    INF = 10**12
    dist = {sid: 0.0 for sid in src_ids}
    pq = [(0.0, sid) for sid in src_ids]
    heapq.heapify(pq)
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return pd.DataFrame({"station_id": list(dist.keys()), "time_sec_station": list(dist.values())})


def compute_commute_minutes_from(origin_name: str, g_polys: gpd.GeoDataFrame) -> pd.DataFrame:
    K_NEAREST_ST = 5
    WALK_KMPH = 5.0
    WALK_M_PER_MIN = (WALK_KMPH * 1000) / 60.0
    MAX_WALK_M = 15000
    SOFTMIN_TAU_MIN = 8.0
    VMAX_GLOBAL_KMH = 160.0
    BOARD_MIN = 4.0

    def _ensure_lv95(gdf):
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_CH)
        if str(gdf.crs).lower() != "epsg:2056":
            gdf = gdf.to_crs(CRS_CH)
        return gdf

    g_polys = _ensure_lv95(g_polys)

    seed_ids = match_station_ids_by_name(origin_name)
    if not seed_ids:
        return g_polys[["GEMEINDE_CODE"]].assign(avg_travel_min=np.nan)

    dist_df = _multi_source_dijkstra(seed_ids)
    st_pts = stations_pts.merge(dist_df, on="station_id", how="left")
    valid = st_pts.dropna(subset=["time_sec_station"]).copy()
    if valid.empty:
        return g_polys[["GEMEINDE_CODE"]].assign(avg_travel_min=np.nan)

    # in-polygon
    st_in = gpd.sjoin(
        valid[["station_id", "time_sec_station", "geometry"]],
        g_polys[["GEMEINDE_CODE", "geometry"]],
        how="left",
        predicate="within",
    ).dropna(subset=["GEMEINDE_CODE"])
    inpoly = st_in.groupby("GEMEINDE_CODE", as_index=False)["time_sec_station"].mean()
    inpoly["avg_travel_min"] = inpoly["time_sec_station"] / 60.0
    inpoly = inpoly[["GEMEINDE_CODE", "avg_travel_min"]].copy()

    # soft-min via K-nearest (rail + walk)
    g_cent = g_polys.copy()
    g_cent["centroid"] = g_cent.geometry.representative_point()
    g_cent = g_cent.set_geometry("centroid")

    remaining = g_cent.copy()
    cand = valid[["station_id", "time_sec_station", "geometry"]].copy()
    nearest = []
    for k in range(K_NEAREST_ST):
        kn = gpd.sjoin_nearest(remaining, cand, how="left", distance_col=f"dist_m_{k}")
        cols = {}
        if "GEMEINDE_CODE_left" in kn.columns:
            cols["GEMEINDE_CODE_left"] = "GEMEINDE_CODE"
        if "time_sec_station_right" in kn.columns:
            cols["time_sec_station_right"] = f"time_sec_{k}"
        if "station_id_right" in kn.columns:
            cols["station_id_right"] = f"sid_{k}"
        kn = kn.rename(columns=cols)
        if f"time_sec_{k}" not in kn.columns and "time_sec_station" in kn.columns:
            kn = kn.rename(columns={"time_sec_station": f"time_sec_{k}"})
        if f"sid_{k}" not in kn.columns and "station_id" in kn.columns:
            kn = kn.rename(columns={"station_id": f"sid_{k}"})
        nearest.append(kn[["GEMEINDE_CODE", f"sid_{k}", f"time_sec_{k}", f"dist_m_{k}"]].copy())
        matched = kn.get(f"sid_{k}", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
        if matched:
            cand = cand[~cand["station_id"].astype(str).isin(matched)].copy()

    knn = nearest[0]
    for tdf in nearest[1:]:
        knn = knn.merge(tdf, on="GEMEINDE_CODE", how="left")

    tot_cols = []
    for k in range(K_NEAREST_ST):
        dcol, tcol = f"dist_m_{k}", f"time_sec_{k}"
        if dcol in knn.columns and tcol in knn.columns:
            knn[dcol] = np.minimum(knn[dcol].fillna(np.inf), MAX_WALK_M)
            walk_min = knn[dcol] / WALK_M_PER_MIN
            rail_min = knn[tcol] / 60.0
            knn[f"total_min_{k}"] = rail_min + walk_min
            tot_cols.append(f"total_min_{k}")

    if tot_cols:
        arr = knn[tot_cols].to_numpy(dtype="float64")
        arr = np.where(np.isfinite(arr), arr, np.inf)
        Z = np.exp(-np.clip(arr, 0, np.inf) / SOFTMIN_TAU_MIN)
        Z[arr == np.inf] = 0.0
        soft = -SOFTMIN_TAU_MIN * np.log(np.clip(Z.sum(axis=1), 1e-9, np.inf))
        knn_out = pd.DataFrame({"GEMEINDE_CODE": knn["GEMEINDE_CODE"].values, "softmin_min": soft})
    else:
        knn_out = pd.DataFrame(
            {"GEMEINDE_CODE": knn["GEMEINDE_CODE"].values, "softmin_min": np.nan}
        )

    # combine + physics floor + origin zeroing
    comb = g_polys[["GEMEINDE_CODE", "geometry"]].merge(inpoly, on="GEMEINDE_CODE", how="left")
    comb = comb.merge(knn_out, on="GEMEINDE_CODE", how="left")
    comb["avg_travel_min"] = comb["avg_travel_min"].fillna(comb["softmin_min"]).astype(float)
    comb = comb.drop(columns=["softmin_min"])

    seeds_geom = stations_pts.loc[stations_pts["station_id"].isin(seed_ids), "geometry"]
    if len(seeds_geom) == 0:
        seeds_geom = stations_pts.iloc[:1]["geometry"]
    origin_pt = _uu(list(seeds_geom)).centroid

    rep_pts = comb["geometry"].representative_point()
    dist_m = rep_pts.distance(origin_pt).to_numpy(dtype="float64")
    floor_min = 4.0 + (dist_m / 1000.0) / VMAX_GLOBAL_KMH * 60.0
    comb["avg_travel_min"] = np.maximum(comb["avg_travel_min"].to_numpy(dtype="float64"), floor_min)

    origin_code = g_polys.loc[g_polys.contains(origin_pt), "GEMEINDE_CODE"]
    if not origin_code.empty:
        comb.loc[comb["GEMEINDE_CODE"] == origin_code.iloc[0], "avg_travel_min"] = 0.0

    # light Gaussian smoothing (optional: skip here for speed)
    return comb[["GEMEINDE_CODE", "avg_travel_min"]].copy()
