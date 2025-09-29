#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Builds app/data/* artifacts from local inputs:
- data/vacancy_municipality.csv      (BFS-style vacancy rates per municipality)
- data/gtfs_train.zip                (GTFS bundle)

Outputs (primary):
- data/artifacts/gemeinden.geojson
- data/artifacts/gemeinden_simplified.geojson
- data/artifacts/gemeinden_centroids.parquet
- data/artifacts/gemeinden.csv
- data/artifacts/meta.json
- app/data/{gemeinden.geojson, gemeinden_simplified.geojson, meta.json}

Optional convenience (single-origin table so the app shows a selector):
- app/data/tt_by_origin.csv

Run:
  .venv/Scripts/python scripts/make_artifacts.py
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

# --- local package imports
from sch.geo_utils import CRS_CH, CRS_WGS84, norm_name, safe_to_crs
from sch.gtfs_graph import build_station_graph, compute_commute_minutes_from
from sch.io_utils import parse_percent, read_bfs_like_csv
from sch.scoring import blend_preference

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
ART = DATA / "artifacts"
APP_DATA = ROOT / "app" / "data"
ART.mkdir(parents=True, exist_ok=True)
APP_DATA.mkdir(parents=True, exist_ok=True)

# Inputs can be either in data/ or data/raw/
VACANCY_CSV = (
    (DATA / "vacancy_municipality.csv")
    if (DATA / "vacancy_municipality.csv").exists()
    else (RAW / "vacancy_municipality.csv")
)
GTFS_ZIP = (
    (DATA / "gtfs_train.zip") if (DATA / "gtfs_train.zip").exists() else (RAW / "gtfs_train.zip")
)

# Default origin for precomputed travel times (also used if dynamic origins are missing)
ORIGIN_STATION = "Zürich HB"

# Cache for geometries (avoid re-downloading each run)
GEM_CACHE = DATA / "gemeinden_cache.geojson"


# ---------- Helpers ----------
def log(msg: str) -> None:
    print(msg, flush=True)


def sanity_inputs() -> None:
    log("▶ Sanity check for input files")
    for path, label in [(VACANCY_CSV, "vacancy_municipality.csv"), (GTFS_ZIP, "gtfs_train.zip")]:
        status = "OK" if Path(path).exists() else "MISSING"
        print(f"  • {Path(path).resolve()} : {status}")
    assert VACANCY_CSV.exists(), f"Missing file: {VACANCY_CSV}"
    assert GTFS_ZIP.exists(), f"Missing file: {GTFS_ZIP}"


def load_gemeinden_geo() -> gpd.GeoDataFrame:
    """Download Swiss Gemeinde polygons (latest vintage) → CH CRS."""
    log("▶ Loading Gemeinde geometries …")
    if GEM_CACHE.exists():
        log(f"  → using cached {GEM_CACHE}")
        gdf_all = gpd.read_file(GEM_CACHE)
    else:
        log("▶ Downloading Gemeinde geometries …")
        # Public dataset with vintages; we filter to latest.
        url = (
            "https://data.opendatasoft.com/explore/dataset/"
            "georef-switzerland-gemeinde-millesime%40public/download/"
            "?format=geojson&timezone=Europe%2FBerlin"
        )
        gdf_all = gpd.read_file(url)[
            ["gem_name", "gem_code", "kan_code", "year", "geometry"]
        ].rename(
            columns={
                "gem_name": "GEMEINDE_NAME",
                "gem_code": "GEMEINDE_CODE",
                "kan_code": "KANTON_CODE",
            }
        )
        try:
            # Write cache as proper GeoJSON (UTF-8, no BOM)
            gdf_all.to_file(GEM_CACHE, driver="GeoJSON")
        except Exception as e:
            log(f"  ! could not cache geometries: {e}")

    latest_year = gdf_all["year"].max()
    gdf = (
        gdf_all[gdf_all["year"] == latest_year]
        .sort_values(["GEMEINDE_CODE", "year"])
        .drop_duplicates("GEMEINDE_CODE", keep="last")
        .copy()
    )
    gdf = safe_to_crs(gdf.set_geometry(gdf.geometry.buffer(0)), CRS_CH)
    gdf["key_norm"] = gdf["GEMEINDE_NAME"].map(norm_name)
    return gdf


def load_vacancy(csv_path: Path) -> pd.DataFrame:
    """Read BFS-like CSV and extract % vacancy per municipality."""
    log("▶ Reading vacancy CSV …")
    df_raw = read_bfs_like_csv(str(csv_path))

    COL_GEM = "Grossregionen, Kantone, Bezirke und Gemeinden"
    COL_ROOMS = "Anzahl Zimmer"
    COL_TYPE = "Typ der leer stehenden Wohnung"
    COL_MEAS = "Art der Messung"
    COL_VAL = "OBS_VALUE" if "OBS_VALUE" in df_raw.columns else "Beobachtungswert"

    df = df_raw.copy()
    df["key_name"] = df[COL_GEM].astype(str).str.strip()
    m_rooms = df[COL_ROOMS].astype(str).str.strip().str.lower().isin(["total", "_t", "gesamt"])
    m_type = df[COL_TYPE].astype(str).str.lower().str.contains("alle", case=False)
    m_meas = df[COL_MEAS].astype(str).str.lower().str.contains("anteil", case=False)
    df = df[m_rooms & m_type & m_meas].copy()

    df["vacancy_pct"] = df[COL_VAL].map(parse_percent)
    df["key_norm"] = df["key_name"].map(norm_name)
    return df[["key_norm", "vacancy_pct"]]


def unzip_gtfs(gtfs_zip: Path) -> Path:
    log("▶ Unzipping GTFS …")
    gtfs_dir = DATA / "gtfs"
    if gtfs_dir.exists():
        log("  • GTFS directory already exists")
        return gtfs_dir
    gtfs_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gtfs_zip, "r") as z:
        z.extractall(gtfs_dir)
    return gtfs_dir


def load_gtfs_tables(
    gtfs_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("▶ Reading GTFS tables …")
    stops = pd.read_csv(gtfs_dir / "stops.txt")
    trips = pd.read_csv(gtfs_dir / "trips.txt")
    routes = pd.read_csv(gtfs_dir / "routes.txt")
    stop_times = pd.read_csv(gtfs_dir / "stop_times.txt")
    return stops, trips, routes, stop_times


# ---------- Main pipeline ----------
def main() -> None:
    sanity_inputs()

    # 1) Load geometries & vacancy and join
    g = load_gemeinden_geo()
    ind = load_vacancy(VACANCY_CSV)
    log("▶ Joining vacancy → Gemeinden …")
    g = g.merge(ind, on="key_norm", how="left")

    # 2) GTFS → commute times
    gtfs_dir = unzip_gtfs(GTFS_ZIP)
    stops, trips, routes, stop_times = load_gtfs_tables(gtfs_dir)

    log("▶ Building station graph …")
    # build_station_graph returns dict with e.g. adj, stations_pts, stops, st_times
    graph_globals = build_station_graph(stops, trips, routes, stop_times)

    # Make available to compute function if it looks up globals()
    globals().update(graph_globals)

    log(f"▶ Computing commute from origin: {ORIGIN_STATION}")
    g_m = g.to_crs(CRS_CH)[["GEMEINDE_CODE", "geometry"]].copy()
    tt = compute_commute_minutes_from(ORIGIN_STATION, g_m)
    tt["GEMEINDE_CODE"] = tt["GEMEINDE_CODE"].astype(str)
    g["GEMEINDE_CODE"] = g["GEMEINDE_CODE"].astype(str)

    g = g.merge(tt, on="GEMEINDE_CODE", how="left").rename(
        columns={"avg_travel_min": "avg_travel_min"}
    )

    # 3) Preference score (precompute a reasonable baseline)
    log("▶ Computing preference score …")
    score, meta_weights = blend_preference(
        g["vacancy_pct"],
        g["avg_travel_min"],
        a=2.0,
        b=2.0,
        k=3.0,  # fixed a,b; k only adjustable in the app
    )
    g["preference_score"] = score

    # 4) Export artifacts
    log("▶ Exporting artifacts …")
    export_cols = [
        "GEMEINDE_CODE",
        "GEMEINDE_NAME",
        "vacancy_pct",
        "avg_travel_min",
        "preference_score",
        "geometry",
    ]
    g_export = g.to_crs(CRS_WGS84)[export_cols].copy()

    # Full GeoJSON
    (ART / "gemeinden.geojson").unlink(missing_ok=True)
    g_export.to_file(ART / "gemeinden.geojson", driver="GeoJSON")
    log(f"  → wrote {ART / 'gemeinden.geojson'}")

    # Simplified GeoJSON (lighter for cloud)
    g_simplified = g_export.copy()
    g_simplified["geometry"] = g_simplified.geometry.simplify(0.0012, preserve_topology=True)
    (ART / "gemeinden_simplified.geojson").unlink(missing_ok=True)
    g_simplified.to_file(ART / "gemeinden_simplified.geojson", driver="GeoJSON")
    log(f"  → wrote {ART / 'gemeinden_simplified.geojson'}")

    # Centroids (useful for some visualizations)
    cent = g_export.copy()
    cent["geometry"] = cent.geometry.representative_point()
    cent.to_parquet(ART / "gemeinden_centroids.parquet", index=False)
    log(f"  → wrote {ART / 'gemeinden_centroids.parquet'}")

    # CSV without geometry
    g_export.drop(columns=["geometry"]).to_csv(ART / "gemeinden.csv", index=False)
    log(f"  → wrote {ART / 'gemeinden.csv'}")

    # Meta.json
    meta_out = {
        "origin_station": ORIGIN_STATION,
        # keep k here as a default; a,b not user-adjustable in the app
        "penalty_k": 1.5,
        "canonical_weights": {
            "w0": float(meta_weights["w0"]),
            "a": 2.0,
            "b": 2.0,
        },
    }
    with open(ART / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    log(f"  → wrote {ART / 'meta.json'}")

    # 5) Copy a few files for the app
    for fname in ["gemeinden.geojson", "gemeinden_simplified.geojson", "meta.json"]:
        src = ART / fname
        dst = APP_DATA / fname
        dst.unlink(missing_ok=True)
        src.replace(dst)
    log(f"  → synced app data to {APP_DATA}")

    # 6) Optional: emit a minimal tt_by_origin.csv so the UI shows a selector
    #    (One origin only; you can replace with multi-origin precomputation later.)
    tto = tt[["GEMEINDE_CODE", "avg_travel_min"]].copy()
    tto["origin_name"] = ORIGIN_STATION
    tto = tto[["origin_name", "GEMEINDE_CODE", "avg_travel_min"]]
    tto.to_csv(APP_DATA / "tt_by_origin.csv", index=False)
    log(f"  → wrote {APP_DATA / 'tt_by_origin.csv'}")

    log("✅ Done.")


if __name__ == "__main__":
    main()
