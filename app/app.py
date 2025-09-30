# app/app.py
# -*- coding: utf-8 -*-

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Swiss Housing & Commute Explorer", layout="wide")

# ---------------- Utils ----------------


def robust_range(series_like, lo_q=1, hi_q=99):
    s = pd.to_numeric(pd.Series(series_like).astype("float64"), errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return (0.0, 1.0)
    lo = float(np.nanpercentile(s, lo_q))
    hi = float(np.nanpercentile(s, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    return lo, hi


def norm01_clipped(x, lo, hi):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype="float64")
    lo, hi = float(lo), float(hi)
    if hi <= lo + 1e-9:
        return np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def commute_penalty_shape(t_norm, k):
    t = np.clip(np.asarray(t_norm, dtype="float64"), 0.0, 1.0)
    kk = max(0.2, float(k))
    return 1.0 - np.power(1.0 - t, kk)


def enforce_origin_zero_and_shift(
    df_wgs84: gpd.GeoDataFrame, tt_sel: pd.DataFrame | None
) -> gpd.GeoDataFrame:
    df = df_wgs84.copy()
    df["GEMEINDE_CODE"] = df["GEMEINDE_CODE"].astype(str)
    if "avg_travel_min" not in df.columns:
        df["avg_travel_min"] = np.nan
    df["avg_travel_min_eff"] = pd.to_numeric(df["avg_travel_min"], errors="coerce")

    if tt_sel is not None and not tt_sel.empty:
        tmp = tt_sel[["GEMEINDE_CODE", "avg_travel_min"]].copy()
        tmp["GEMEINDE_CODE"] = tmp["GEMEINDE_CODE"].astype(str)
        tmp = tmp.rename(columns={"avg_travel_min": "avg_travel_min_origin"})
        df = df.merge(tmp, on="GEMEINDE_CODE", how="left")
        origin_vals = pd.to_numeric(df["avg_travel_min_origin"], errors="coerce")
        base_vals = pd.to_numeric(df["avg_travel_min_eff"], errors="coerce")
        df["avg_travel_min_eff"] = np.where(np.isfinite(origin_vals), origin_vals, base_vals)

    cur = pd.to_numeric(df["avg_travel_min_eff"], errors="coerce")
    if np.isfinite(cur).any():
        mmin = float(np.nanmin(cur))
        df.loc[cur == mmin, "avg_travel_min_eff"] = 0.0
        df.loc[cur != mmin, "avg_travel_min_eff"] = (cur - mmin)[cur != mmin]
    df["avg_travel_min_eff"] = pd.to_numeric(df["avg_travel_min_eff"], errors="coerce").clip(
        lower=0
    )
    return df


def answers_signature(choices_dict):
    key_order = sorted(choices_dict.keys())
    s = "|".join(f"{k}:{choices_dict[k]}" for k in key_order)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_geojson_robust(path: Path) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(str(path))
    except Exception:
        pass
    try:
        return gpd.read_file(str(path), engine="fiona")
    except Exception:
        pass
    raw = path.read_bytes()
    if len(raw) < 50:
        raise ValueError(f"GeoJSON looks empty: {path}")
    gj = json.loads(raw.decode("utf-8", errors="ignore"))
    if not isinstance(gj, dict) or gj.get("type") != "FeatureCollection":
        raise ValueError(f"Not a FeatureCollection: {path}")
    feats = gj.get("features", [])
    if not feats:
        raise ValueError(f"No features in: {path}")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


@st.cache_data(show_spinner=False)
def load_data():
    gj = DATA_DIR / "gemeinden_simplified.geojson"
    mj = DATA_DIR / "meta.json"
    if not gj.exists() or not mj.exists():
        st.error("Artifacts not found. Run scripts/make_artifacts.py first.")
        st.stop()

    g = read_geojson_robust(gj)
    if g.crs is None or str(g.crs).lower() != "epsg:4326":
        g = g.to_crs(4326)
    for c in ["vacancy_pct", "avg_travel_min", "preference_score"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    meta = json.loads(mj.read_text(encoding="utf-8"))

    tto = None
    ttp = DATA_DIR / "tt_by_origin.parquet"
    if ttp.exists():
        try:
            tto = pd.read_parquet(ttp)
            tto["GEMEINDE_CODE"] = tto["GEMEINDE_CODE"].astype(str)
            tto["avg_travel_min"] = pd.to_numeric(tto["avg_travel_min"], errors="coerce")
            if "origin_name" not in tto.columns and "origin_station_id" in tto.columns:
                tto["origin_name"] = tto["origin_station_id"].astype(str)
        except Exception:
            tto = None

    return g, meta, tto


# ---------------- App ----------------

g, meta, tt_by_origin = load_data()

st.title("Swiss Housing & Commute Explorer")
st.markdown(
    "Explore Swiss municipalities by combining **housing vacancy** and **commute time**. "
    "Three modes: **Preference score**, **Housing only**, **Commute only**. "
    "Pick the **origin station** in the left sidebar."
)

with st.sidebar:
    st.header("Controls")

    def _is_bahn2000(name: str) -> bool:
        s = str(name).lower()
        return ("bahn" in s) and ("2000" in s)

    if (
        tt_by_origin is not None
        and "origin_name" in tt_by_origin.columns
        and tt_by_origin["origin_name"].notna().any()
    ):
        stations_raw = tt_by_origin["origin_name"].dropna().astype(str).unique().tolist()
        stations = sorted([s for s in stations_raw if not _is_bahn2000(s)], key=str.casefold)
        default_origin = meta.get("origin_station", "Zürich HB")
        idx = stations.index(default_origin) if default_origin in stations else 0
        origin_name = st.selectbox("Origin SBB station", options=stations, index=idx)
    else:
        origin_name = meta.get("origin_station", "Zürich HB")
        st.info(f"Dynamic origins not found; using precomputed origin: **{origin_name}**")

    map_mode = st.radio("Map mode", ["Preference score", "Housing only", "Commute only"])
    penalty_k = st.slider(
        "Commute penalty curvature (k)", 0.2, 6.0, float(meta.get("penalty_k", 1.5)), 0.1
    )

df = g.copy()
df["GEMEINDE_CODE"] = df["GEMEINDE_CODE"].astype(str)

if (
    tt_by_origin is not None
    and "origin_name" in tt_by_origin.columns
    and origin_name in tt_by_origin["origin_name"].astype(str).unique()
):
    tt_sel = tt_by_origin.loc[
        tt_by_origin["origin_name"].astype(str).eq(origin_name), ["GEMEINDE_CODE", "avg_travel_min"]
    ].copy()
else:
    tt_sel = None

df = enforce_origin_zero_and_shift(df, tt_sel)

# -> simplify polygons a bit for speed
try:
    df["geometry"] = df.geometry.simplify(0.0005, preserve_topology=True)
except Exception:
    pass

v_range = robust_range(df.get("vacancy_pct", np.nan))
t_range = robust_range(df.get("avg_travel_min_eff", np.nan))


# ---- Preference elicitation scenarios ----
def build_edge_tradeoffs(_df, v_range, t_range, n=10, seed=42):
    if isinstance(_df, (pd.DataFrame, gpd.GeoDataFrame)):
        dfx = _df.copy()
    else:
        dfx = pd.DataFrame(_df)
    need = ["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min"]
    for c in need:
        if c not in dfx.columns:
            dfx[c] = np.nan
    dfx = dfx[need].copy()
    dfx["vacancy_pct"] = pd.to_numeric(dfx["vacancy_pct"], errors="coerce")
    dfx["avg_travel_min"] = pd.to_numeric(dfx["avg_travel_min"], errors="coerce")
    dfx = dfx.dropna(subset=["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min"])
    if dfx.empty:
        return pd.DataFrame(
            columns=[
                "qid",
                "A_gem",
                "A_vacancy_pct",
                "A_travel_min",
                "B_gem",
                "B_vacancy_pct",
                "B_travel_min",
            ]
        )

    v_lo, v_hi = v_range
    t_lo, t_hi = t_range
    dfx["v_n"] = norm01_clipped(dfx["vacancy_pct"].values, v_lo, v_hi)
    dfx["t_n"] = norm01_clipped(dfx["avg_travel_min"].values, t_lo, t_hi)

    rng = np.random.default_rng(seed)
    used = set()
    rows = []
    tries = 0
    while len(rows) < n and tries < 8000:
        tries += 1
        A = dfx.sample(1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
        if A["GEMEINDE_NAME"] in used:
            continue
        orientation = int(rng.integers(0, 2))
        if orientation == 0:
            cand = dfx[
                (dfx["v_n"] < A["v_n"])
                & (dfx["t_n"] < A["t_n"])
                & ((A["v_n"] - dfx["v_n"]).between(0.10, 0.30, inclusive="both"))
                & ((A["t_n"] - dfx["t_n"]).between(0.05, 0.18, inclusive="both"))
                & (~dfx["GEMEINDE_NAME"].isin(used | {A["GEMEINDE_NAME"]}))
            ]
        else:
            cand = dfx[
                (dfx["v_n"] > A["v_n"])
                & (dfx["t_n"] > A["t_n"])
                & ((dfx["v_n"] - A["v_n"]).between(0.10, 0.30, inclusive="both"))
                & ((dfx["t_n"] - A["t_n"]).between(0.05, 0.18, inclusive="both"))
                & (~dfx["GEMEINDE_NAME"].isin(used | {A["GEMEINDE_NAME"]}))
            ]
        if cand.empty:
            continue
        B = cand.sample(1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
        rows.append(
            {
                "qid": len(rows) + 1,
                "A_gem": A["GEMEINDE_NAME"],
                "A_vacancy_pct": float(A["vacancy_pct"]),
                "A_travel_min": float(A["avg_travel_min"]),
                "B_gem": B["GEMEINDE_NAME"],
                "B_vacancy_pct": float(B["vacancy_pct"]),
                "B_travel_min": float(B["avg_travel_min"]),
            }
        )
        used.add(A["GEMEINDE_NAME"])
        used.add(B["GEMEINDE_NAME"])
    return pd.DataFrame(rows)


df_scen = df[["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min_eff"]].copy()
df_scen = df_scen.rename(columns={"avg_travel_min_eff": "avg_travel_min"})
sc = build_edge_tradeoffs(df_scen, v_range=v_range, t_range=t_range, n=10, seed=42)

tab_map, tab_pref, tab_data = st.tabs(["🗺️ Map", "🧭 Preference elicitation", "📄 Data"])

with tab_pref:
    st.subheader("10 trade-off questions")
    if sc.empty:
        st.warning("Not enough data to build scenarios.")
    else:
        for _, row in sc.iterrows():
            st.markdown("---")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown(f"**Q{int(row.qid)} – Option A**")
                st.write(row.A_gem)
                st.write(f"Vacancy: {row.A_vacancy_pct:.2f}%")
                st.write(f"Commute: {row.A_travel_min:.1f} min")
            with c2:
                st.markdown(f"**Q{int(row.qid)} – Option B**")
                st.write(row.B_gem)
                st.write(f"Vacancy: {row.B_vacancy_pct:.2f}%")
                st.write(f"Commute: {row.B_travel_min:.1f} min")
            with c3:
                st.radio(
                    f"Your choice for Q{int(row.qid)}",
                    ["A", "B"],
                    key=f"choice_{int(row.qid)}",
                    horizontal=True,
                    index=None,
                )

# record simple signature to avoid refit loops (future extension)
choices_dict, answered_rows = {}, []
for _, r in sc.iterrows():
    key = f"choice_{int(r.qid)}"
    ch = st.session_state.get(key, None)
    if ch in ("A", "B"):
        choices_dict[key] = ch
        answered_rows.append(r)
st.session_state["answers_sig"] = answers_signature(choices_dict) if choices_dict else None

with tab_data:
    show = df.copy()
    show["avg_travel_min_eff"] = pd.to_numeric(show["avg_travel_min_eff"], errors="coerce")
    cols = ["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min_eff", "preference_score"]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(
        show[cols].rename(columns={"avg_travel_min_eff": "avg_travel_min"}),
        use_container_width=True,
    )
    st.caption(f"Total rows: {len(show)}")

with tab_map:
    if map_mode == "Housing only":
        metric_col = "vacancy_pct"
        legend = "Vacancy rate (%)"
        colors = ["red", "yellow", "green"]
        vals = pd.to_numeric(df["vacancy_pct"], errors="coerce")
        vmin = float(np.nanpercentile(vals, 1)) if vals.notna().any() else 0.0
        vmax = float(np.nanpercentile(vals, 99)) if vals.notna().any() else 1.0
        df_render = df.copy()

    elif map_mode == "Commute only":
        metric_col = "avg_travel_min_eff"
        legend = "Commute time from origin (min)"
        colors = ["green", "yellow", "red"]
        vals = pd.to_numeric(df["avg_travel_min_eff"], errors="coerce")
        vmin = 0.0
        vmax = float(np.nanpercentile(vals, 99)) if vals.notna().any() else 120.0
        df_render = df.dropna(subset=[metric_col]).copy()

    else:
        metric_col = "preference_score"
        legend = "Preference score (0–100)"
        colors = ["red", "yellow", "green"]
        # dynamic recompute allowed but we already embedded a default one; keep it simple:
        vmin, vmax = 0.0, 100.0
        df_render = df.dropna(subset=["vacancy_pct", "avg_travel_min_eff"]).copy()
        if (
            "preference_score" not in df_render.columns
            or df_render["preference_score"].isna().all()
        ):
            # compute on the fly if missing
            v_lo, v_hi = v_range
            t_lo, t_hi = t_range
            v_all = norm01_clipped(
                pd.to_numeric(df_render["vacancy_pct"], errors="coerce").values, v_lo, v_hi
            )
            t_all = norm01_clipped(
                pd.to_numeric(df_render["avg_travel_min_eff"], errors="coerce").values, t_lo, t_hi
            )
            pen_t = commute_penalty_shape(t_all, penalty_k=float(meta.get("penalty_k", 1.5)))
            util = 0.0 + 2.0 * v_all - 2.0 * pen_t
            finite = np.isfinite(util)
            if finite.any():
                util = util - np.median(util[finite])
            df_render["preference_score"] = 100.0 * (1.0 / (1.0 + np.exp(-util)))

    # Folium rendering
    def render_folium(df_map):
        import branca.colormap as bcm
        import folium
        import streamlit.components.v1 as components
        from streamlit_folium import st_folium

        colormap = bcm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
        colormap.caption = legend
        m = folium.Map(location=[46.8, 8.2], zoom_start=7, tiles="OpenStreetMap")

        if metric_col == "vacancy_pct":
            df_map["vacancy_round"] = pd.to_numeric(df_map["vacancy_pct"], errors="coerce").round(2)
            fields, aliases = ["GEMEINDE_NAME", "vacancy_round"], ["Municipality", "Vacancy (%)"]
        elif metric_col == "avg_travel_min_eff":
            df_map["time_round"] = pd.to_numeric(
                df_map["avg_travel_min_eff"], errors="coerce"
            ).round(1)
            fields, aliases = ["GEMEINDE_NAME", "time_round"], ["Municipality", "Commute (min)"]
        else:
            df_map["score_round"] = pd.to_numeric(
                df_map["preference_score"], errors="coerce"
            ).round(1)
            df_map["vacancy_round"] = pd.to_numeric(df_map["vacancy_pct"], errors="coerce").round(2)
            df_map["time_round"] = pd.to_numeric(
                df_map["avg_travel_min_eff"], errors="coerce"
            ).round(1)
            fields, aliases = ["GEMEINDE_NAME", "score_round", "vacancy_round", "time_round"], [
                "Municipality",
                "Score",
                "Vacancy (%)",
                "Commute (min)",
            ]

        folium.GeoJson(
            data=df_map.to_json(),
            style_function=lambda feat: {
                "fillColor": (
                    colormap(float(feat["properties"].get(metric_col)))
                    if feat["properties"].get(metric_col) is not None
                    else "#cccccc"
                ),
                "color": "white",
                "weight": 0.2,
                "fillOpacity": 0.85,
            },
            tooltip=folium.features.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
        ).add_to(m)
        colormap.add_to(m)
        try:
            st_folium(m, height=720, width=None)
        except Exception:
            html = m._repr_html_()
            components.html(html, height=720, scrolling=False)

    if df_render.empty:
        st.warning("No municipalities available to render for this selection.")
    else:
        render_folium(df_render)
