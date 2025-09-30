# scripts/make_artifacts.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
APP_DATA = ROOT / "app" / "data"
ARTIFACTS = DATA / "artifacts"

# Inputs you already have/expect
VACANCY_CSV = DATA / "vacancy_municipality.csv"  # BFS SDMX export (semicolon)
GEMEINDEN_CACHE = DATA / "gemeinden_cache.geojson"  # local cache (recommended)

# If the cache is missing, we’ll auto-download a clean snapshot once:
OD_URL = (
    "https://data.opendatasoft.com/explore/dataset/"
    "georef-switzerland-gemeinde-millesime%40public/download/?format=geojson"
)

# ------------------------------- utils ---------------------------------------


def ensure_dirs():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    APP_DATA.mkdir(parents=True, exist_ok=True)


def _read_text_any(path: Path) -> str | None:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return None


def read_bfs_sdmx_csv(path: Path) -> pd.DataFrame:
    """
    Robust reader for BFS SDMX-style CSV (semicolon; header not always first line).
    """
    txt = _read_text_any(path)
    if txt is None:
        raise IOError(f"Cannot read file: {path}")
    lines = txt.splitlines()
    hdr_pat = re.compile(r"(TIME_PERIOD|Zeitperiode).*(OBS_VALUE|Beobachtungswert)", re.I)
    hdr_idx = 0
    for i, ln in enumerate(lines[:200]):
        if hdr_pat.search(ln):
            hdr_idx = i
            break
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr_idx:])), sep=";", engine="python", dtype=str)
    return df


def parse_percent(x):
    """'0,51' → 0.51 ; '51' → 51.0 (we will not divide by 100 here; BFS already in %)"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("\xa0", " ").replace("%", "").strip()
    s = s.replace("’", "").replace("'", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def norm_name(s: str) -> str:
    import unicodedata as ucn

    if pd.isna(s):
        return ""
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
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()


def norm01(x, lo=None, hi=None):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype="float64")
    f = x[np.isfinite(x)]
    if lo is None or hi is None:
        if f.size == 0:
            return np.zeros_like(x)
        lo = float(np.nanpercentile(f, 1))
        hi = float(np.nanpercentile(f, 99))
        if hi <= lo:
            lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
            if hi <= lo:
                hi = lo + 1.0
    lo, hi = float(lo), float(hi)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-12)


def commute_penalty(t_norm: np.ndarray, k: float) -> np.ndarray:
    t = np.clip(np.asarray(t_norm, dtype="float64"), 0.0, 1.0)
    kk = max(0.2, float(k))
    return 1.0 - np.power(1.0 - t, kk)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# -------------------------- geometry loading ---------------------------------


def _download_gemeinden_if_needed() -> Path:
    if GEMEINDEN_CACHE.exists():
        return GEMEINDEN_CACHE
    print("▶ Gemeinden cache missing — downloading once from Opendatasoft …")
    import urllib.request

    DATA.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(OD_URL) as resp:
        raw = resp.read()
    GEMEINDEN_CACHE.write_bytes(raw)
    return GEMEINDEN_CACHE


def load_gemeinden_geo() -> gpd.GeoDataFrame:
    p = _download_gemeinden_if_needed()
    gdf_all = gpd.read_file(p)[["gem_name", "gem_code", "kan_code", "year", "geometry"]].rename(
        columns={
            "gem_name": "GEMEINDE_NAME",
            "gem_code": "GEMEINDE_CODE",
            "kan_code": "KANTON_CODE",
        }
    )
    latest = gdf_all["year"].max()
    g = (
        gdf_all[gdf_all["year"] == latest]
        .sort_values(["GEMEINDE_CODE", "year"])
        .drop_duplicates("GEMEINDE_CODE", keep="last")
        .copy()
    )
    if g.crs is None or str(g.crs).lower() != "epsg:4326":
        g = g.to_crs(4326)
    g["GEMEINDE_CODE"] = g["GEMEINDE_CODE"].astype(str)
    return g


# ------------------------- vacancy parsing/join -------------------------------


def load_vacancy_latest() -> tuple[pd.DataFrame, dict]:
    if not VACANCY_CSV.exists():
        raise FileNotFoundError(f"Missing vacancy CSV: {VACANCY_CSV}")
    df = read_bfs_sdmx_csv(VACANCY_CSV)

    # try both DE/EN label variants seen in BFS exports
    C_NAME = next((c for c in df.columns if "Grossregionen" in c or "Gemeinden" in c), None)
    C_CODE = "GR_KT_GDE" if "GR_KT_GDE" in df.columns else None
    C_MEAS = "MEASURE_DIMENSION" if "MEASURE_DIMENSION" in df.columns else None
    C_TYPE = "LEERWOHN_TYP" if "LEERWOHN_TYP" in df.columns else None
    C_VAL = (
        "OBS_VALUE"
        if "OBS_VALUE" in df.columns
        else ("Beobachtungswert" if "Beobachtungswert" in df.columns else None)
    )
    C_YEAR = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else None

    req = [C_NAME, C_VAL, C_YEAR]
    if any(x is None for x in req):
        raise KeyError(f"Vacancy CSV missing expected columns; got: {list(df.columns)}")

    # filter: measurement=PC (percent), type=_T (total) when present
    sel = df.copy()
    if C_MEAS and C_MEAS in sel.columns:
        sel = sel[sel[C_MEAS].astype(str).str.upper().eq("PC")]
    if C_TYPE and C_TYPE in sel.columns:
        sel = sel[sel[C_TYPE].astype(str).str.upper().eq("_T")]

    years = pd.to_numeric(sel[C_YEAR], errors="coerce")
    latest = int(np.nanmax(years))
    sel = sel[years.eq(latest)].copy()

    sel["key_name"] = sel[C_NAME].astype(str).str.strip()
    sel["vacancy_pct"] = sel[C_VAL].map(parse_percent)

    if C_CODE and C_CODE in sel.columns:
        sel["GEMEINDE_CODE"] = sel[C_CODE].astype(str)
    else:
        sel["GEMEINDE_CODE"] = np.nan  # will rely on name join

    out = sel[["GEMEINDE_CODE", "key_name", "vacancy_pct"]].copy()
    meta = {"vacancy_year": latest, "source": "BFS SDMX (PC, _T)"}
    return out, meta


def join_vacancy(g: gpd.GeoDataFrame, vac: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Join vacancy onto Gemeinde polygons:
      1) by BFS code (authoritative if present)
      2) fill remaining via normalized name match
    Always guarantees 'vacancy_pct' exists in g.
    """
    g = g.copy()
    vac = vac.copy()

    # Ensure the target column exists before any merges
    if "vacancy_pct" not in g.columns:
        g["vacancy_pct"] = np.nan

    # 1) by code
    if "GEMEINDE_CODE" in vac.columns and vac["GEMEINDE_CODE"].notna().any():
        vac_code = vac.dropna(subset=["GEMEINDE_CODE"]).copy()
        vac_code["GEMEINDE_CODE"] = vac_code["GEMEINDE_CODE"].astype(str)
        vac_code = vac_code[["GEMEINDE_CODE", "vacancy_pct"]].rename(
            columns={"vacancy_pct": "vacancy_pct_code"}
        )
        g = g.merge(vac_code, on="GEMEINDE_CODE", how="left")
        # prefer code-based values where available
        if "vacancy_pct_code" in g.columns:
            g["vacancy_pct"] = g["vacancy_pct"].combine_first(g["vacancy_pct_code"])
            g.drop(columns=["vacancy_pct_code"], inplace=True)

    # 2) fill remaining by normalized names
    miss = g["vacancy_pct"].isna()
    if miss.any():
        tmp = vac.copy()
        tmp["key_norm"] = tmp["key_name"].map(norm_name)
        g["key_norm"] = g["GEMEINDE_NAME"].map(norm_name)
        filled = g.loc[miss, ["key_norm"]].merge(
            tmp[["key_norm", "vacancy_pct"]], on="key_norm", how="left"
        )
        g.loc[miss, "vacancy_pct"] = filled["vacancy_pct"].values
        g.drop(columns=["key_norm"], inplace=True, errors="ignore")

    # clean up known non-municipals if present
    non_exact = {
        "Zürichsee (ZH)",
        "Thunersee",
        "Brienzersee",
        "Bielersee (BE)",
        "Bielersee (NE)",
        "Lac de Neuchâtel (BE)",
        "Lac de Neuchâtel (NE)",
        "Bodensee (SG)",
        "Bodensee (TG)",
        "Staatswald Galm",
    }
    mask_non = g["GEMEINDE_NAME"].isin(non_exact) | g["GEMEINDE_NAME"].str.contains(
        r"^Comunanza\b", case=False, na=False
    )
    g.loc[mask_non, "vacancy_pct"] = np.nan

    return g


# -------------------------- distance-based travel -----------------------------

DEFAULT_ORIGINS: Dict[str, tuple[float, float]] = {
    "Zürich HB": (47.378177, 8.540192),
    "Bern": (46.948824, 7.439132),
    "Basel SBB": (47.547451, 7.589626),
    "Genève": (46.210206, 6.142439),
    "Lausanne": (46.516003, 6.629099),
    "Luzern": (47.050168, 8.310229),
    "St. Gallen": (47.423180, 9.369775),
    "Winterthur": (47.499346, 8.724128),
    "Zug": (47.172423, 8.517376),
    "Biel/Bienne": (47.136669, 7.246791),
    "Fribourg/Freiburg": (46.806477, 7.161971),
    "Neuchâtel": (46.989987, 6.929273),
    "Sion": (46.227388, 7.360625),
    "Sierre/Siders": (46.294000, 7.535000),
    "Brig": (46.319232, 7.988532),
    "Visp": (46.293000, 7.881000),
    "Chur": (46.853000, 9.529000),
    "Bellinzona": (46.195000, 9.029000),
    "Lugano": (46.004000, 8.950000),
    "Arth-Goldau": (47.048000, 8.545000),
    "Thun": (46.754000, 7.629000),
    "Interlaken Ost": (46.690000, 7.869000),
    "Schaffhausen": (47.697000, 8.633000),
    "Solothurn": (47.207000, 7.537000),
    "Aarau": (47.392000, 8.044000),
    "Baden": (47.476000, 8.308000),
    "Wil SG": (47.460000, 9.050000),
    "Rapperswil SG": (47.226000, 8.817000),
    "La Chaux-de-Fonds": (47.104000, 6.826000),
    "Delémont": (47.363000, 7.344000),
}


def compute_distance_based_tt(
    g: gpd.GeoDataFrame,
    origins: Dict[str, tuple[float, float]],
    cruise_kmh: float = 50.0,
    overhead_min: float = 10.0,
) -> pd.DataFrame:
    """Always produces a full table for all Gemeinden × origins (no NaNs)."""
    if g.crs is None or str(g.crs).lower() != "epsg:4326":
        g = g.to_crs(4326)
    cent = g.geometry.representative_point()
    lat = cent.y.values
    lon = cent.x.values

    rows = []
    for oname, (olat, olon) in origins.items():
        dist_km = np.array(
            [haversine_km(olat, olon, lat[i], lon[i]) for i in range(len(g))], dtype="float64"
        )
        tt_min = overhead_min + (dist_km / max(cruise_kmh, 1.0)) * 60.0
        rows.append(
            pd.DataFrame(
                {
                    "origin_name": oname,
                    "GEMEINDE_CODE": g["GEMEINDE_CODE"].astype(str).values,
                    "avg_travel_min": tt_min,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------- main --------------------------------------


def build_artifacts(origins_csv: str, default_origin: str, penalty_k: float) -> None:
    ensure_dirs()

    print("▶ Loading Gemeinde polygons …")
    g = load_gemeinden_geo()
    print(f"  → {len(g)} municipalities")

    print("▶ Parsing vacancy CSV …")
    vac, vmeta = load_vacancy_latest()
    print(f"  → latest year: {vmeta.get('vacancy_year')}")

    print("▶ Joining vacancy …")
    g = join_vacancy(g, vac)
    print(
        f"  → vacancy coverage: {g['vacancy_pct'].notna().mean():.3f} "
        f"({g['vacancy_pct'].notna().sum()}/{len(g)})"
    )

    # Origins list
    if origins_csv.strip():
        requested = [s.strip() for s in origins_csv.split(";") if s.strip()]
    else:
        requested = list(DEFAULT_ORIGINS.keys())
    origins = {name: DEFAULT_ORIGINS.get(name, DEFAULT_ORIGINS["Zürich HB"]) for name in requested}

    print("▶ Computing distance-based travel times …")
    tt_by_origin = compute_distance_based_tt(g, origins)
    if default_origin not in origins:
        default_origin = requested[0]
    tdef = tt_by_origin[tt_by_origin["origin_name"] == default_origin][
        ["GEMEINDE_CODE", "avg_travel_min"]
    ].copy()

    # Attach default travel time to polygons
    g_out = g.merge(tdef, on="GEMEINDE_CODE", how="left")

    # Embed simple preference score (immediate mapability)
    v_n = norm01(g_out["vacancy_pct"])
    t_n = norm01(g_out["avg_travel_min"])
    pen = commute_penalty(t_n, penalty_k)
    w0, a, b = 0.0, 2.0, 2.0
    util = w0 + a * v_n - b * pen
    util = util - np.nanmedian(util[np.isfinite(util)])
    g_out["preference_score"] = 100.0 * (1.0 / (1.0 + np.exp(-util)))

    # Simplify geometry for small file size
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

    print("▶ Exporting artifacts …")
    simplified_path = ARTIFACTS / "gemeinden_simplified.geojson"
    g_simpl.to_file(simplified_path, driver="GeoJSON")
    print(f"  → wrote {simplified_path}")

    # Sync to app/data
    APP_DATA.mkdir(parents=True, exist_ok=True)
    (APP_DATA / "gemeinden_simplified.geojson").write_text(
        simplified_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print("  → copied to app/data/gemeinden_simplified.geojson")

    # Multi-origin matrix
    tt_parquet = APP_DATA / "tt_by_origin.parquet"
    tt_by_origin.to_parquet(tt_parquet, index=False)
    print(
        f"  → wrote {tt_parquet} ({len(tt_by_origin):,} rows, {tt_by_origin['origin_name'].nunique()} origins)"
    )

    # Meta
    meta = {
        "origin_station": default_origin,
        "penalty_k": float(penalty_k),
        "vacancy_year": vmeta.get("vacancy_year"),
        "canonical_weights": {"w0": 0.0, "a": 2.0, "b": 2.0},
    }
    (APP_DATA / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("  → wrote app/data/meta.json")
    print("✅ Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--origins",
        type=str,
        default=";".join(DEFAULT_ORIGINS.keys()),
        help="Semicolon-separated list of origin station names (must be keys in DEFAULT_ORIGINS).",
    )
    p.add_argument("--default-origin", type=str, default="Zürich HB")
    p.add_argument("--penalty-k", type=float, default=1.5)
    return p.parse_args()


def main():
    args = parse_args()
    build_artifacts(args.origins, args.default_origin, args.penalty_k)


if __name__ == "__main__":
    main()
