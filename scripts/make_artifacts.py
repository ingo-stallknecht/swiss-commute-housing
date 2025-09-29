# scripts/make_artifacts.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
APP_DATA = ROOT / "app" / "data"
ARTIFACTS = DATA / "artifacts"

VACANCY_CSV = DATA / "vacancy_municipality.csv"
GEMEINDEN_CACHE = DATA / "gemeinden_cache.geojson"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def ensure_dirs():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    APP_DATA.mkdir(parents=True, exist_ok=True)


def robust_read_csv(path: Path) -> pd.DataFrame:
    """Try default CSV; if a single mega-column, re-read as semicolon SDMX."""
    df = pd.read_csv(path, engine="python")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=";", engine="python")
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance (km)."""
    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def norm01(x, lo=None, hi=None):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype="float64")
    if lo is None or hi is None:
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return np.zeros_like(x)
        lo = float(np.nanpercentile(finite, 1))
        hi = float(np.nanpercentile(finite, 99))
        if hi <= lo:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            if hi <= lo:
                hi = lo + 1.0
    lo, hi = float(lo), float(hi)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-12)


def commute_penalty(t_norm: np.ndarray, k: float) -> np.ndarray:
    t = np.clip(np.asarray(t_norm, dtype="float64"), 0.0, 1.0)
    kk = max(0.2, float(k))
    return 1.0 - np.power(1.0 - t, kk)


# ------------------------------------------------------------
# Load Gemeinde polygons
# ------------------------------------------------------------


def load_gemeinden_geo() -> gpd.GeoDataFrame:
    """
    Load Gemeinde polygons from local cache. If you need to (re)create this
    cache from OGD, do it once locally and commit only the simplified artifact.
    """
    if not GEMEINDEN_CACHE.exists():
        raise FileNotFoundError(
            f"Missing {GEMEINDEN_CACHE}. Create it once locally (download OGD polygons) "
            "or copy an existing cache into data/."
        )
    g = gpd.read_file(GEMEINDEN_CACHE)
    if g.crs is None or str(g.crs).lower() != "epsg:4326":
        g = g.to_crs(4326)

    # Try to normalize ID & name columns
    # Common OGD schema: 'GEMEINDE_CODE' (BFS code), 'GEMEINDE_NAME'
    candidates_id = [
        "GEMEINDE_CODE",
        "BFS",
        "BFSNR",
        "GMDNR",
        "bfs",
        "GR_KT_GDE",  # last resort (present in your vacancy CSV too)
    ]
    candidates_name = ["GEMEINDE_NAME", "NAME", "NAME_DE", "gemname"]

    g_cols = {c.lower(): c for c in g.columns}
    gid = next((g_cols[c.lower()] for c in candidates_id if c.lower() in g_cols), None)
    gname = next((g_cols[c.lower()] for c in candidates_name if c.lower() in g_cols), None)

    if gid is None:
        raise KeyError(
            "Could not find a Gemeinde code column in polygons. Expect one of "
            f"{candidates_id}. Got: {list(g.columns)}"
        )
    if gname is None:
        # not fatal; we can proceed without it
        gname = gid

    g = g.rename(columns={gid: "GEMEINDE_CODE", gname: "GEMEINDE_NAME"})
    g["GEMEINDE_CODE"] = g["GEMEINDE_CODE"].astype(str)

    return g


# ------------------------------------------------------------
# Vacancy loader (handles BFS SDMX)
# ------------------------------------------------------------


def _load_vacancy() -> Tuple[pd.DataFrame, dict]:
    """
    Parse BFS SDMX export (semicolon-separated) and extract vacancy percentage
    per municipality for the latest available year.

    Uses:
      - code: GR_KT_GDE
      - name: Grossregionen, Kantone, Bezirke und Gemeinden
      - measure: MEASURE_DIMENSION == 'PC'
      - type: LEERWOHN_TYP == '_T'
      - value: OBS_VALUE
      - year: TIME_PERIOD (take latest)
    """
    if not VACANCY_CSV.exists():
        raise FileNotFoundError(f"Missing vacancy CSV: {VACANCY_CSV}")

    df = robust_read_csv(VACANCY_CSV)
    cols = list(df.columns)
    print("▶ Vacancy CSV columns:", cols)

    # Detect columns we need
    col_code = "GR_KT_GDE"  # present in your probe
    col_name = "Grossregionen, Kantone, Bezirke und Gemeinden"
    col_meas = "MEASURE_DIMENSION"
    col_type = "LEERWOHN_TYP"
    col_val = "OBS_VALUE"
    col_year = "TIME_PERIOD"

    for req in [col_code, col_name, col_meas, col_type, col_val, col_year]:
        if req not in df.columns:
            raise KeyError(f"Vacancy file missing '{req}'. Present columns: {list(df.columns)}")

    # Filter PC (percentage), total type, latest year
    sel = df.copy()
    sel = sel[sel[col_meas].astype(str).str.upper().eq("PC")]
    sel = sel[sel[col_type].astype(str).str.upper().eq("_T")]
    years = pd.to_numeric(sel[col_year], errors="coerce")
    latest = int(years.max())
    sel = sel[years.eq(latest)].copy()

    # Build result
    sel["GEMEINDE_CODE"] = sel[col_code].astype(str)
    sel["GEMEINDE_NAME"] = sel[col_name].astype(str)
    sel["vacancy_pct"] = pd.to_numeric(sel[col_val], errors="coerce")

    out = (
        sel[["GEMEINDE_CODE", "GEMEINDE_NAME", "vacancy_pct"]]
        .groupby(["GEMEINDE_CODE", "GEMEINDE_NAME"], as_index=False)
        .mean()
    )
    meta = {"vacancy_year": latest, "source": "BFS SDMX (PC, _T)"}
    return out, meta


# ------------------------------------------------------------
# Travel times (fallback: distance-based)
# ------------------------------------------------------------

DEFAULT_ORIGINS = {
    "Zürich HB": (47.378177, 8.540192),
    "Bern": (46.948824, 7.439132),
    "Basel SBB": (47.547451, 7.589626),
    "Genève": (46.210206, 6.142439),
    "Lausanne": (46.516003, 6.629099),
    "Luzern": (47.050168, 8.310229),
    "St. Gallen": (47.423180, 9.369775),
    "Winterthur": (47.499346, 8.724128),
}


def compute_distance_based_tt(
    g: gpd.GeoDataFrame,
    origins: dict[str, tuple[float, float]],
    cruise_kmh: float = 50.0,
    overhead_min: float = 10.0,
) -> pd.DataFrame:
    """
    Fallback travel times: straight-line distance / speed + overhead.
    Produces a tidy table: [origin_name, GEMEINDE_CODE, avg_travel_min].
    """
    if g.crs is None or str(g.crs).lower() != "epsg:4326":
        g = g.to_crs(4326)
    cent = g.geometry.centroid
    lat = cent.y.values
    lon = cent.x.values

    rows = []
    for oname, (olat, olon) in origins.items():
        dist_km = np.array(
            [haversine_km(olat, olon, lat[i], lon[i]) for i in range(len(g))], dtype="float64"
        )
        tt_min = overhead_min + (dist_km / max(cruise_kmh, 1.0)) * 60.0
        tmp = pd.DataFrame(
            {
                "origin_name": oname,
                "GEMEINDE_CODE": g["GEMEINDE_CODE"].astype(str).values,
                "avg_travel_min": tt_min,
            }
        )
        rows.append(tmp)

    tto = pd.concat(rows, ignore_index=True)
    return tto


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def build_artifacts(
    origins_csv: str,
    default_origin: str,
    penalty_k: float,
    only_simplified: bool = True,
) -> None:
    ensure_dirs()

    print("▶ Checking inputs …")
    print(f"  • {VACANCY_CSV}: {'OK' if VACANCY_CSV.exists() else 'MISSING'}")
    print(f"  • {GEMEINDEN_CACHE}: {'OK (local cache)' if GEMEINDEN_CACHE.exists() else 'MISSING'}")

    g = load_gemeinden_geo()
    print(f"▶ Loading Gemeinde polygons …\n  → {len(g)} polygons")

    v, vmeta = _load_vacancy()
    print("▶ Read vacancy …")

    # Join vacancy onto polygons (prefer code; fallback name)
    g["GEMEINDE_CODE"] = g["GEMEINDE_CODE"].astype(str)
    v["GEMEINDE_CODE"] = v["GEMEINDE_CODE"].astype(str)
    g = g.merge(v[["GEMEINDE_CODE", "vacancy_pct"]], on="GEMEINDE_CODE", how="left")

    # Origins
    if origins_csv.strip():
        origins_list = [s.strip() for s in origins_csv.split(";") if s.strip()]
    else:
        origins_list = list(DEFAULT_ORIGINS.keys())

    # Make sure we have coordinates for the requested origins (use defaults when missing)
    origins = {}
    for o in origins_list:
        if o in DEFAULT_ORIGINS:
            origins[o] = DEFAULT_ORIGINS[o]
        else:
            # Use Zürich HB coords as a generic fallback
            origins[o] = DEFAULT_ORIGINS["Zürich HB"]

    # Compute travel times (fallback model)
    print("▶ Computing distance-based travel times …")
    tt_by_origin = compute_distance_based_tt(g, origins)

    # Pick default origin times to embed in the simplified file
    if default_origin not in origins:
        default_origin = origins_list[0]
    tdef = tt_by_origin[tt_by_origin["origin_name"] == default_origin][
        ["GEMEINDE_CODE", "avg_travel_min"]
    ].copy()

    g_out = g.merge(tdef, on="GEMEINDE_CODE", how="left")

    # Preference score embedded (so the app can render immediately)
    v_n = norm01(g_out["vacancy_pct"])
    t_n = norm01(g_out["avg_travel_min"])
    pen = commute_penalty(t_n, penalty_k)
    # Simple fixed weights; app can still re-weigh interactively
    w0, a, b = 0.0, 2.0, 2.0
    util = w0 + a * v_n - b * pen
    util = util - np.nanmedian(util[np.isfinite(util)])
    g_out["preference_score"] = 100.0 * (1.0 / (1.0 + np.exp(-util)))

    # Simplify geometry for a small artifact
    try:
        g_simpl = g_out.to_crs(3857)
        g_simpl["geometry"] = g_simpl.geometry.simplify(200, preserve_topology=True)
        g_simpl = g_simpl.to_crs(4326)
    except Exception:
        g_simpl = g_out.copy()

    keep = [
        "GEMEINDE_CODE",
        "GEMEINDE_NAME",
        "vacancy_pct",
        "avg_travel_min",
        "preference_score",
        "geometry",
    ]
    g_simpl = g_simpl[keep]

    # Write artifacts
    print("▶ Exporting artifacts …")
    simplified_path = ARTIFACTS / "gemeinden_simplified.geojson"
    g_simpl.to_file(simplified_path, driver="GeoJSON")
    print(f"  → wrote {simplified_path}")

    # Copy into app/data
    (APP_DATA / "gemeinden_simplified.geojson").write_text(
        simplified_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print("  → copied gemeinden_simplified.geojson to app/data/")

    # tt_by_origin (for multiple origins in the app)
    tt_parquet = APP_DATA / "tt_by_origin.parquet"
    tt_by_origin.to_parquet(tt_parquet, index=False)
    print(f"  → wrote {tt_parquet}")

    # meta
    meta = {
        "origin_station": default_origin,
        "penalty_k": penalty_k,
        "vacancy_year": vmeta.get("vacancy_year"),
        "canonical_weights": {"w0": 0.0, "a": 2.0, "b": 2.0},
    }
    (APP_DATA / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("  → wrote app/data/meta.json")

    print("✅ Done. Artifacts in:", ARTIFACTS)
    print("✅ App data synced to:", APP_DATA)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--origins",
        type=str,
        default="Zürich HB;Bern;Basel SBB;Genève;Lausanne;Luzern;St. Gallen;Winterthur",
    )
    p.add_argument("--default-origin", type=str, default="Zürich HB")
    p.add_argument("--penalty-k", type=float, default=1.5)
    p.add_argument(
        "--only-simplified",
        action="store_true",
        default=True,
        help="Kept for compatibility; this builder always writes only simplified artifacts.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    build_artifacts(
        origins_csv=args.origins,
        default_origin=args.default_origin,
        penalty_k=args.penalty_k,
        only_simplified=True,
    )


if __name__ == "__main__":
    main()
