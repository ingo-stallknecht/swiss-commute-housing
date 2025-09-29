#!/usr/bin/env python
from __future__ import annotations
import os, json, zipfile
from pathlib import Path
import pandas as pd
import geopandas as gpd

from sch.io_utils import read_bfs_like_csv, parse_percent
from sch.geo_utils import safe_to_crs, norm_name, CRS_CH, CRS_WGS84
from sch.scoring import blend_preference
from sch.gtfs_graph import build_station_graph, compute_commute_minutes_from

ROOT = Path(__file__).resolve().parents[1]

# ---- PATHS -------------------------------------------------------------------
DATA = ROOT / "data"
ART = DATA / "artifacts"
APP_DATA = ROOT / "app" / "data"
ART.mkdir(parents=True, exist_ok=True)
APP_DATA.mkdir(parents=True, exist_ok=True)

VACANCY_CSV = DATA / "vacancy_municipality.csv"
GTFS_ZIP    = DATA / "gtfs_train.zip"

ORIGIN_STATION = "Zürich HB"

# ---- HELPERS -----------------------------------------------------------------
def load_gemeinden_geo(cache_path: Path) -> gpd.GeoDataFrame:
    URL_GEM = ("https://data.opendatasoft.com/explore/dataset/"
               "georef-switzerland-gemeinde-millesime%40public/download/"
               "?format=geojson&timezone=Europe%2FBerlin")
    if cache_path.exists():
        gdf_all = gpd.read_file(cache_path)
    else:
        gdf_all = gpd.read_file(URL_GEM)
        gdf_all.to_file(cache_path, driver="GeoJSON")
    gdf_all = gdf_all[["gem_name","gem_code","kan_code","year","geometry"]].rename(
        columns={"gem_name":"GEMEINDE_NAME","gem_code":"GEMEINDE_CODE","kan_code":"KANTON_CODE"}
    )
    latest_geom_year = gdf_all["year"].max()
    gdf = (gdf_all[gdf_all["year"] == latest_geom_year]
           .sort_values(["GEMEINDE_CODE","year"])
           .drop_duplicates("GEMEINDE_CODE", keep="last")
           .copy())
    gdf = safe_to_crs(gdf.set_geometry(gdf.geometry.buffer(0)), CRS_CH)
    return gdf

def load_vacancy(path_csv: Path) -> pd.DataFrame:
    df_raw = read_bfs_like_csv(str(path_csv))
    COL_GEM   = "Grossregionen, Kantone, Bezirke und Gemeinden"
    COL_ROOMS = "Anzahl Zimmer"
    COL_TYPE  = "Typ der leer stehenden Wohnung"
    COL_MEAS  = "Art der Messung"
    COL_VAL   = "OBS_VALUE" if "OBS_VALUE" in df_raw.columns else "Beobachtungswert"

    df = df_raw.copy()
    df["key_name"] = df[COL_GEM].astype(str).str.strip()
    m_rooms = df[COL_ROOMS].astype(str).str.strip().str.lower().isin(["total","_t","gesamt"])
    m_type  = df[COL_TYPE ].astype(str).str.lower().str.contains("alle", case=False)
    m_meas  = df[COL_MEAS ].astype(str).str.lower().str.contains("anteil", case=False)
    df = df[m_rooms & m_type & m_meas].copy()
    df["vacancy_pct"] = df[COL_VAL].map(parse_percent)
    df["key_norm"] = df["key_name"].map(norm_name)
    return df[["key_norm","vacancy_pct"]]

def ensure_station_id(stops: pd.DataFrame) -> pd.DataFrame:
    s = stops.copy()
    if "station_id" not in s.columns:
        parent = s.get("parent_station")
        if parent is None:
            parent = pd.Series([None]*len(s), index=s.index)
        s["station_id"] = parent.fillna(s["stop_id"]).astype(str)
    return s

# ---- MAIN --------------------------------------------------------------------
def main():
    print("▶ Sanity check for input files")
    print(f"  • {VACANCY_CSV} : {'OK' if VACANCY_CSV.exists() else 'MISSING'}")
    print(f"  • {GTFS_ZIP}    : {'OK' if GTFS_ZIP.exists() else 'MISSING'}")
    assert VACANCY_CSV.exists(), f"Missing file: {VACANCY_CSV}"
    assert GTFS_ZIP.exists(),    f"Missing file: {GTFS_ZIP}"

    print("▶ Loading Gemeinde geometries …")
    cache_gj = DATA / "gemeinden_cache.geojson"
    g = load_gemeinden_geo(cache_gj)
    g["key_norm"] = g["GEMEINDE_NAME"].map(norm_name)

    print("▶ Reading vacancy CSV …")
    ind = load_vacancy(VACANCY_CSV)

    print("▶ Joining vacancy → Gemeinden …")
    g = g.merge(ind, on="key_norm", how="left")

    print("▶ Unzipping GTFS …")
    gtfs_dir = DATA / "gtfs"
    if not gtfs_dir.exists():
        with zipfile.ZipFile(GTFS_ZIP, "r") as z:
            z.extractall(gtfs_dir)
    else:
        print("  • GTFS directory already exists")

    print("▶ Reading GTFS tables …")
    stops_df      = pd.read_csv(gtfs_dir / "stops.txt")
    trips         = pd.read_csv(gtfs_dir / "trips.txt")
    routes        = pd.read_csv(gtfs_dir / "routes.txt")
    stop_times_df = pd.read_csv(gtfs_dir / "stop_times.txt")
    # pre-ensure station_id for robustness (gtfs_graph will do it again, harmless)
    stops_df = ensure_station_id(stops_df)

    print("▶ Building station graph …")
    globals().update(build_station_graph(stops_df, trips, routes, stop_times_df))

    print(f"▶ Computing commute from origin: {ORIGIN_STATION}")
    g_m = g.to_crs(CRS_CH)[["GEMEINDE_CODE","geometry"]].copy()
    tt = compute_commute_minutes_from(ORIGIN_STATION, g_m)
    tt["GEMEINDE_CODE"] = tt["GEMEINDE_CODE"].astype(str)
    g["GEMEINDE_CODE"] = g["GEMEINDE_CODE"].astype(str)
    g = g.merge(tt, on="GEMEINDE_CODE", how="left")

    print("▶ Computing preference score …")
    score, meta = blend_preference(g["vacancy_pct"], g["avg_travel_min"], a=2.0, b=2.0, k=3.0)
    g["preference_score"] = score

    print("▶ Exporting artifacts …")
    export_cols = ["GEMEINDE_CODE","GEMEINDE_NAME","vacancy_pct","avg_travel_min","preference_score","geometry"]
    g_export = g.to_crs(CRS_WGS84)[export_cols].copy()

    g_export.to_file(ART / "gemeinden.geojson", driver="GeoJSON")
    g_slim = g_export.copy()
    g_slim["geometry"] = g_slim.geometry.simplify(0.0012, preserve_topology=True)
    g_slim.to_file(ART / "gemeinden_simplified.geojson", driver="GeoJSON")
    cent = g_export.copy()
    cent["geometry"] = cent.geometry.representative_point()
    cent.to_parquet(ART / "gemeinden_centroids.parquet", index=False)
    g_export.drop(columns=["geometry"]).to_csv(ART / "gemeinden.csv", index=False)
    print(f"  → wrote {ART/'gemeinden.geojson'}")
    print(f"  → wrote {ART/'gemeinden_simplified.geojson'}")
    print(f"  → wrote {ART/'gemeinden_centroids.parquet'}")
    print(f"  → wrote {ART/'gemeinden.csv'}")

    # Stations lookup (SAFE: ensure station_id/stop_name exist)
    stops_for_lookup = ensure_station_id(stops_df)
    if "stop_name" not in stops_for_lookup.columns:
        # some feeds use 'stop_desc' or similar; fall back gracefully
        stops_for_lookup["stop_name"] = stops_for_lookup.get("stop_desc", "unknown")
    station_choices = (stops_for_lookup[["station_id","stop_name"]]
                       .dropna().drop_duplicates().sort_values("stop_name"))
    station_choices = station_choices[~station_choices["stop_name"].str.contains("Bahn-2000", case=False, na=False)]
    station_choices.to_csv(ART / "stations_lookup.csv", index=False)
    print(f"  → wrote {ART/'stations_lookup.csv'}")

    # ---- Multi-origin travel time table (drives origin dropdown) --------------
    print("▶ Building multi-origin tt_by_origin (top 20 + curated extras)")
    by_station = (st_times.groupby("station_id", as_index=False)["stop_id"]
                  .count().rename(columns={"stop_id":"events"})
                  .sort_values("events", ascending=False))
    top20 = by_station.head(20).copy()
    name_map = (stops_for_lookup.groupby("station_id")["stop_name"]
                .agg(lambda s: s.value_counts().idxmax()))
    top20["origin_name"] = top20["station_id"].map(name_map.to_dict())

    extras = ["Neuchâtel","Fribourg/Freiburg","Sion","Lugano","Bellinzona","Chur"]
    # match by best-effort normalization
    def _normtxt(s: str) -> str:
        s = str(s).lower()
        for a,b in (("ä","a"),("ö","o"),("ü","u"),("é","e"),("è","e"),("ê","e")):
            s = s.replace(a,b)
        return s.strip()
    # reverse index for matching
    reverse = { _normtxt(v): k for k,v in name_map.items() if isinstance(v,str) }
    extra_rows = []
    for nm in extras:
        sid = reverse.get(_normtxt(nm))
        if sid and sid not in set(top20["station_id"].astype(str)):
            extra_rows.append({"station_id": sid, "origin_name": name_map.get(sid, nm)})
    if extra_rows:
        top20 = pd.concat([top20[["station_id","origin_name"]], pd.DataFrame(extra_rows)], ignore_index=True)
    top20 = top20.dropna(subset=["origin_name"]).drop_duplicates("station_id").reset_index(drop=True)

    rows = []
    for _, r in top20.iterrows():
        origin_name = str(r["origin_name"])
        res = compute_commute_minutes_from(origin_name, g_m[["GEMEINDE_CODE","geometry"]].copy())
        res["origin_name"] = origin_name
        rows.append(res[["GEMEINDE_CODE","avg_travel_min","origin_name"]])

    if rows:
        tt_by_origin = pd.concat(rows, ignore_index=True)
        tt_by_origin["GEMEINDE_CODE"] = tt_by_origin["GEMEINDE_CODE"].astype(str)
        tt_by_origin["avg_travel_min"] = pd.to_numeric(tt_by_origin["avg_travel_min"], errors="coerce")
        tt_by_origin.to_parquet(ART / "tt_by_origin.parquet", index=False)
        print(f"  → wrote {ART/'tt_by_origin.parquet'} ({len(tt_by_origin):,} rows, "
              f"{tt_by_origin['origin_name'].nunique()} origins)")
        # copy to app/data
        (ART / "tt_by_origin.parquet").replace(APP_DATA / "tt_by_origin.parquet")
        print(f"  → copied to {APP_DATA/'tt_by_origin.parquet'}")
    else:
        print("  ⚠️ Could not build tt_by_origin (no candidate origins).")

    meta_out = {
        "origin_station": ORIGIN_STATION,
        **meta
    }
    with open(ART / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    (ART / "meta.json").replace(APP_DATA / "meta.json")
    (ART / "gemeinden.geojson").replace(APP_DATA / "gemeinden.geojson")

    print("✅ Done. Artifacts in:", ART)
    print("✅ App data synced to:", APP_DATA)

if __name__ == "__main__":
    main()
