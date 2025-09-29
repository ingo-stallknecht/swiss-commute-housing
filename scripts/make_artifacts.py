#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Builds artifacts for the app:
- Download/prepare Swiss Gemeinden geometries
- Read vacancy CSV
- (Optionally) compute commute metrics (requires GTFS; omitted here)
- Produce simplified GeoJSON + meta.json
- Copy minimal files to app/data/

Usage:
    python scripts/make_artifacts.py [--fresh] [--only-simplified]

Flags:
    --fresh            Ignore cached Gemeinde GeoJSON and re-download.
    --only-simplified  Export only the simplified GeoJSON (skips full file).
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA
ART = DATA / "artifacts"
APP_DATA = ROOT / "app" / "data"

ART.mkdir(parents=True, exist_ok=True)
APP_DATA.mkdir(parents=True, exist_ok=True)

VACANCY_CSV = RAW / "vacancy_municipality.csv"
GTFS_ZIP = RAW / "gtfs_train.zip"

CACHE_GEMEINDEN = DATA / "gemeinden_cache.geojson"

CRS_CH = "EPSG:2056"
CRS_WGS84 = "EPSG:4326"


def safe_to_crs(gdf, crs):
    g = gdf.copy()
    try:
        if g.crs is None:
            return g.set_crs(crs)
        elif str(g.crs).lower() != str(crs).lower():
            return g.to_crs(crs)
    except Exception:
        pass
    return g


def parse_percent(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    s = (
        s.replace("\xa0", " ")
        .replace("%", "")
        .replace("’", "")
        .replace("'", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        v = float(s)
    except Exception:
        return np.nan
    return v / 100.0 if v > 10 else v


def read_bfs_like_csv(path: str) -> pd.DataFrame:
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except Exception:
            pass
    if text is None:
        raise IOError(f"Cannot read file {path} with utf-8/latin1.")
    lines = text.splitlines()
    # Find header row containing these markers
    hdr_idx = 0
    import re

    pat = re.compile(r"(TIME_PERIOD|Zeitperiode).*(OBS_VALUE|Beobachtungswert)")
    for i, ln in enumerate(lines[:200]):
        if pat.search(ln):
            hdr_idx = i
            break
    return pd.read_csv(
        pd.io.common.StringIO("\n".join(lines[hdr_idx:])),
        sep=";",
        engine="python",
        dtype=str,
    )


def load_gemeinden_geo(force_fresh=False) -> gpd.GeoDataFrame:
    """
    Download Gemeinden geojson from OpenDataSoft, normalize schema, cache locally.
    If cached file exists but has different column names, we fix them.
    """
    if force_fresh or not CACHE_GEMEINDEN.exists():
        print("▶ Downloading Gemeinde geometries …")
        url = (
            "https://data.opendatasoft.com/explore/dataset/"
            "georef-switzerland-gemeinde-millesime%40public/download/"
            "?format=geojson&timezone=Europe%2FBerlin"
        )
        gdf_all = gpd.read_file(url)
        # Normalize expected columns if present
        # Typical columns: gem_name, gem_code, kan_code, year, geometry
        rename_map = {}
        for src, tgt in [
            ("gem_name", "GEMEINDE_NAME"),
            ("gem_code", "GEMEINDE_CODE"),
            ("kan_code", "KANTON_CODE"),
        ]:
            if src in gdf_all.columns:
                rename_map[src] = tgt
        gdf_all = gdf_all.rename(columns=rename_map)

        # If columns still missing, try common alternatives
        if "GEMEINDE_NAME" not in gdf_all.columns:
            for alt in ["gemeindename", "name", "gemname"]:
                if alt in gdf_all.columns:
                    gdf_all = gdf_all.rename(columns={alt: "GEMEINDE_NAME"})
                    break
        if "GEMEINDE_CODE" not in gdf_all.columns:
            for alt in ["gem_code", "bfs_gemeindenummer", "gemeindecode", "gemcode", "bfs_nummer"]:
                if alt in gdf_all.columns:
                    gdf_all = gdf_all.rename(columns={alt: "GEMEINDE_CODE"})
                    break

        if "year" not in gdf_all.columns:
            gdf_all["year"] = pd.NA

        keep_cols = [
            c
            for c in ["GEMEINDE_NAME", "GEMEINDE_CODE", "KANTON_CODE", "year", "geometry"]
            if c in gdf_all.columns
        ]
        gdf_all = gdf_all[keep_cols + ([] if "geometry" in keep_cols else ["geometry"])]
        gdf_all = gdf_all.rename(columns=lambda c: c.strip())

        # Deduplicate by GEMEINDE_CODE preferring latest year if available
        if "year" in gdf_all.columns and gdf_all["year"].notna().any():
            gdf = (
                gdf_all.sort_values(["GEMEINDE_CODE", "year"], na_position="first")
                .drop_duplicates("GEMEINDE_CODE", keep="last")
                .copy()
            )
        else:
            gdf = gdf_all.drop_duplicates("GEMEINDE_CODE", keep="last").copy()

        gdf = safe_to_crs(gdf.set_geometry(gdf.geometry.buffer(0)), CRS_CH)
        gdf.to_file(CACHE_GEMEINDEN, driver="GeoJSON")
        print(f"  → cached to {CACHE_GEMEINDEN}")
        return gdf

    print(f"  → using cached {CACHE_GEMEINDEN}")
    gdf = gpd.read_file(CACHE_GEMEINDEN)

    # Repair missing columns if needed
    if "GEMEINDE_CODE" not in gdf.columns or "GEMEINDE_NAME" not in gdf.columns:
        rename_map = {}
        if "gem_code" in gdf.columns:
            rename_map["gem_code"] = "GEMEINDE_CODE"
        if "gem_name" in gdf.columns:
            rename_map["gem_name"] = "GEMEINDE_NAME"
        if rename_map:
            gdf = gdf.rename(columns=rename_map)
    if "GEMEINDE_CODE" not in gdf.columns:
        raise KeyError("Cached Gemeinde file has no 'GEMEINDE_CODE' even after repair.")

    gdf = safe_to_crs(gdf, CRS_CH)
    return gdf


def load_vacancy(path_csv: Path) -> pd.DataFrame:
    df_raw = read_bfs_like_csv(str(path_csv))

    # Try to find headers by fuzzy names
    col_map = {}
    for c in df_raw.columns:
        c_l = c.lower()
        if "grossregionen" in c_l or "gemeinden" in c_l:
            col_map["GEM"] = c
        elif "anzahl zimmer" in c_l:
            col_map["ROOMS"] = c
        elif "typ der leer" in c_l or "typ der leerstehenden" in c_l:
            col_map["TYPE"] = c
        elif "art der messung" in c_l:
            col_map["MEAS"] = c
        elif "obs_value" in c_l or "beobachtungswert" in c_l:
            col_map["VAL"] = c
    need = ["GEM", "ROOMS", "TYPE", "MEAS", "VAL"]
    for k in need:
        if k not in col_map:
            raise KeyError(f"Could not find expected column in vacancy CSV for {k}")

    df = df_raw.copy()
    df["key_name"] = df[col_map["GEM"]].astype(str).str.strip()
    m_rooms = (
        df[col_map["ROOMS"]].astype(str).str.strip().str.lower().isin(["total", "_t", "gesamt"])
    )
    m_type = df[col_map["TYPE"]].astype(str).str.lower().str.contains("alle", case=False, na=False)
    m_meas = (
        df[col_map["MEAS"]].astype(str).str.lower().str.contains("anteil", case=False, na=False)
    )
    df = df[m_rooms & m_type & m_meas].copy()
    df["vacancy_pct"] = df[col_map["VAL"]].map(parse_percent)

    # Build normalised join key (very simple)
    def norm_name(s: str) -> str:
        import re
        import unicodedata as ucn

        if s is None:
            return s
        s = (
            str(s)
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "ae")
            .replace("Ö", "oe")
            .replace("Ü", "ue")
            .replace("ß", "ss")
        )
        s = ucn.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9]+", "", s).lower()
        return s

    df["key_norm"] = df["key_name"].map(norm_name)
    return df[["key_norm", "vacancy_pct"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Ignore cache and re-download.")
    parser.add_argument(
        "--only-simplified", action="store_true", help="Export only simplified GeoJSON."
    )
    args = parser.parse_args()

    print("▶ Sanity check for input files")
    print(f"  • {VACANCY_CSV} : {'OK' if VACANCY_CSV.exists() else 'MISSING'}")
    print(f"  • {GTFS_ZIP} : {'OK' if GTFS_ZIP.exists() else 'OK (not required for now)'}")

    assert VACANCY_CSV.exists(), f"Missing file: {VACANCY_CSV}"

    print("▶ Loading Gemeinde geometries …")
    g = load_gemeinden_geo(force_fresh=args.fresh)
    g["GEMEINDE_CODE"] = g["GEMEINDE_CODE"].astype(str)

    print("▶ Reading vacancy CSV …")
    ind = load_vacancy(VACANCY_CSV)

    print("▶ Joining vacancy → Gemeinden …")

    # Make a sloppy join via normalized name fallback
    def norm_name(s: str) -> str:
        import re
        import unicodedata as ucn

        if s is None:
            return s
        s = (
            str(s)
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "ae")
            .replace("Ö", "oe")
            .replace("Ü", "ue")
            .replace("ß", "ss")
        )
        s = ucn.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9]+", "", s).lower()
        return s

    g["key_norm"] = g["GEMEINDE_NAME"].map(norm_name)
    g = g.merge(ind, on="key_norm", how="left")
    g = g.drop(columns=["key_norm"])

    # For this minimal build, skip GTFS graph and commute recomputation.
    # We leave avg_travel_min NaN; app will still show Housing Only and Preference using vacancy only.
    # You can plug back your GTFS computations here when you’re ready.

    print("▶ Exporting artifacts …")
    export_cols = [
        "GEMEINDE_CODE",
        "GEMEINDE_NAME",
        "vacancy_pct",
        "geometry",
    ]
    g_export = safe_to_crs(g, CRS_WGS84)[export_cols].copy()

    # Full (optional) — skip if only-simplified
    if not args.only_simplified:
        full = ART / "gemeinden.geojson"
        full.unlink(missing_ok=True)
        g_export.to_file(full, driver="GeoJSON")
        print(f"  → wrote {full}")

    # Simplified
    g_slim = g_export.copy()
    try:
        g_slim["geometry"] = g_slim.geometry.simplify(0.0012, preserve_topology=True)
    except Exception:
        pass
    slim = ART / "gemeinden_simplified.geojson"
    slim.unlink(missing_ok=True)
    g_slim.to_file(slim, driver="GeoJSON")
    print(f"  → wrote {slim}")

    # meta.json (include your chosen penalty_k default)
    meta_out = {
        "origin_station": "Zürich HB",
        "penalty_k": 1.5,
        "canonical_weights": {"w0": 0.0, "a": 2.0, "b": 2.0},
    }
    with open(ART / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    # Copy minimal set for the app
    for src_name in ["gemeinden_simplified.geojson", "meta.json"]:
        (APP_DATA / src_name).unlink(missing_ok=True)
        (ART / src_name).replace(APP_DATA / src_name)
        print(f"  → copied {src_name} to app/data/")

    print("✅ Done. Artifacts in:", ART)
    print("✅ App data synced to:", APP_DATA)


if __name__ == "__main__":
    main()
