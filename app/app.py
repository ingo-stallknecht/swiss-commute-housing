# app/app.py
# -*- coding: utf-8 -*-

import codecs
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Setup ----------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Swiss Housing & Commute Explorer", layout="wide")


# ---------- Utils ----------
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
    x = np.asarray(x, dtype="float64")
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
    """Apply origin-specific travel times if provided; otherwise fall back safely.

    Ensures columns exist and handles the case where the GeoJSON has no
    'avg_travel_min' at all (e.g., simplified artifacts). In that case we keep
    avg_travel_min_eff as NaN unless tt_sel provides values.
    """
    df = df_wgs84.copy()

    # Always normalize types and ensure base column exists
    df["GEMEINDE_CODE"] = df["GEMEINDE_CODE"].astype(str)
    if "avg_travel_min" not in df.columns:
        df["avg_travel_min"] = np.nan

    # Start with the base column as the effective one
    df["avg_travel_min_eff"] = pd.to_numeric(df["avg_travel_min"], errors="coerce")

    # If we have origin-specific times, left-merge and prefer those
    if tt_sel is not None and not tt_sel.empty:
        tmp = tt_sel[["GEMEINDE_CODE", "avg_travel_min"]].copy()
        tmp["GEMEINDE_CODE"] = tmp["GEMEINDE_CODE"].astype(str)
        tmp = tmp.rename(columns={"avg_travel_min": "avg_travel_min_origin"})
        df = df.merge(tmp, on="GEMEINDE_CODE", how="left")
        # Prefer origin-specific where available, else keep base
        df["avg_travel_min_eff"] = df["avg_travel_min_origin"].where(
            pd.to_numeric(df["avg_travel_min_origin"], errors="coerce").notna(),
            df["avg_travel_min_eff"],
        )

    # Shift so the best (minimum) becomes 0, if we have any finite values
    cur = pd.to_numeric(df["avg_travel_min_eff"], errors="coerce")
    if np.isfinite(cur).any():
        mmin = float(np.nanmin(cur))
        # 0 for the minimum; subtract min for all others
        df.loc[cur == mmin, "avg_travel_min_eff"] = 0.0
        df.loc[cur != mmin, "avg_travel_min_eff"] = (cur - mmin)[cur != mmin]

    # Final sanitation
    df["avg_travel_min_eff"] = pd.to_numeric(df["avg_travel_min_eff"], errors="coerce").clip(
        lower=0
    )
    return df


def resolve_prior(meta, session):
    if "manual_weights" in session:
        mw = session["manual_weights"]
        return (
            float(mw.get("w0", 0.0)),
            float(max(0.0, mw.get("w_v", 2.0))),
            float(max(0.0, mw.get("w_t", 2.0))),
        )
    try:
        cw = meta["canonical_weights"]
        return (
            float(cw.get("w0", 0.0)),
            float(max(0.0, cw.get("a", 2.0))),
            float(max(0.0, cw.get("b", 2.0))),
        )
    except Exception:
        return (0.0, 2.0, 2.0)


def answers_signature(choices_dict):
    key_order = sorted(choices_dict.keys())
    s = "|".join(f"{k}:{choices_dict[k]}" for k in key_order)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------- Robust GeoJSON loader (handles LFS pointers / HTML) ----------
def _looks_like_lfs_pointer(text_head: str) -> bool:
    head = (text_head or "").strip().lower()
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _looks_like_html(text_head: str) -> bool:
    h = (text_head or "").lstrip().lower()
    return h.startswith("<!doctype html") or h.startswith("<html")


def read_geojson_robust(path: Path) -> gpd.GeoDataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON not found: {p}")

    # 1) Try pyogrio (default in geopandas)
    try:
        return gpd.read_file(str(p))
    except Exception:
        pass

    # 2) Try fiona explicitly
    try:
        return gpd.read_file(str(p), engine="fiona")
    except Exception:
        pass

    # 3) Manual JSON parse with sanity checks
    raw = p.read_bytes()
    if len(raw) < 50:
        raise ValueError(f"GeoJSON looks empty or truncated: {p} (size={len(raw)} bytes)")

    head = raw[:200].decode("utf-8", errors="ignore")
    if _looks_like_lfs_pointer(head):
        raise ValueError(
            f"File appears to be a Git LFS pointer, not actual GeoJSON: {p}.\n"
            f"Fix: untrack with Git LFS or ensure Streamlit Cloud fetches LFS objects."
        )
    if _looks_like_html(head):
        raise ValueError(
            f"File looks like HTML, not GeoJSON: {p}.\n"
            f"Likely a GitHub 'too large to display' page or accidental HTML content."
        )

    try:
        gj = json.loads(raw.decode("utf-8", errors="strict"))
    except Exception as e:
        # fallback encodings
        gj = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                gj = json.loads(raw.decode(enc))
                break
            except Exception:
                pass
        if gj is None:
            raise ValueError(f"Could not parse GeoJSON as JSON: {p}. Error: {e}")

    if not isinstance(gj, dict) or gj.get("type") != "FeatureCollection":
        raise ValueError(f"Not a FeatureCollection: {p}")
    feats = gj.get("features", [])
    if not feats:
        raise ValueError(f"FeatureCollection has no features: {p}")

    try:
        return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    except Exception as e:
        raise ValueError(f"Could not build GeoDataFrame from features: {p}. Error: {e}")


@st.cache_data(show_spinner=False)
def load_data():
    # Prefer smaller/simplified file first for cloud robustness
    simplified = DATA_DIR / "gemeinden_simplified.geojson"
    full = DATA_DIR / "gemeinden.geojson"
    meta_path = DATA_DIR / "meta.json"

    if not meta_path.exists():
        st.error("Artifacts not found. Re-run the export/build step.")
        st.stop()

    # Read meta with encoding safety
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = json.loads(codecs.open(meta_path, "r", "utf-8-sig").read())

    g = None
    errors = []
    for candidate in [simplified, full]:
        if candidate.exists():
            try:
                g = read_geojson_robust(candidate)
                break
            except Exception as e:
                errors.append(f"{candidate.name}: {e}")

    if g is None:
        st.error(
            "Unable to load GeoJSON artifacts.\n\n"
            + "\n".join(f"- {msg}" for msg in errors)
            + "\n\nFix on repo: ensure real GeoJSON is committed (not an LFS pointer or HTML)."
        )
        st.stop()

    if g.crs is None or str(g.crs).lower() != "epsg:4326":
        g = g.to_crs(4326)

    # Coerce known numeric columns if present
    for c in ["vacancy_pct", "avg_travel_min", "preference_score"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    # Ensure base commute column exists (for simplified artifacts)
    if "avg_travel_min" not in g.columns:
        g["avg_travel_min"] = np.nan

    # Try Parquet for dynamic origins; fallback to CSV
    tto = None
    for f in [DATA_DIR / "tt_by_origin.parquet", DATA_DIR / "tt_by_origin.csv"]:
        if f.exists():
            try:
                tto = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                break
            except Exception:
                continue

    if tto is not None:
        if "GEMEINDE_CODE" in tto.columns:
            tto["GEMEINDE_CODE"] = tto["GEMEINDE_CODE"].astype(str)
        if "avg_travel_min" in tto.columns:
            tto["avg_travel_min"] = pd.to_numeric(tto["avg_travel_min"], errors="coerce")
        if "origin_name" not in tto.columns and "origin_station_id" in tto.columns:
            tto["origin_name"] = tto["origin_station_id"].astype(str)

    return g, meta, tto


# ---------- App ----------
g, meta, tt_by_origin = load_data()

st.title("Swiss Housing & Commute Explorer")
st.markdown(
    "Explore Swiss municipalities by combining **housing vacancy** and **commute time**. "
    "Three modes: **Preference score**, **Housing only**, **Commute only**. "
    "Pick the **origin station** in the left sidebar."
)

# ---------- Sidebar ----------
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
        "Commute penalty curvature (k)",
        0.2,
        6.0,
        float(meta.get("penalty_k", 1.5)),
        0.1,
        help="Adjusts the curvature used in Preference score.",
    )

# ---------- Effective commute times ----------
df = g.copy()
df["GEMEINDE_CODE"] = df["GEMEINDE_CODE"].astype(str)

if (
    tt_by_origin is not None
    and origin_name in tt_by_origin.get("origin_name", pd.Series(dtype=str)).astype(str).unique()
):
    tt_sel = tt_by_origin.loc[
        tt_by_origin["origin_name"] == origin_name, ["GEMEINDE_CODE", "avg_travel_min"]
    ].copy()
else:
    tt_sel = None

df = enforce_origin_zero_and_shift(df, tt_sel)

# Simplify for speed (safe no-op if already simple)
try:
    df["geometry"] = df.geometry.simplify(0.0005, preserve_topology=True)
except Exception:
    pass

# ---------- Ranges ----------
v_range = robust_range(df["vacancy_pct"])
t_range = robust_range(df["avg_travel_min_eff"])


# ---------- Scenario scaffolding (edge-case tradeoffs) ----------
def build_edge_tradeoffs(
    _df, v_range, t_range, n=10, seed=42, dtnorm_range=(0.05, 0.18), dvnorm_range=(0.10, 0.30)
):
    if isinstance(_df, (pd.DataFrame, gpd.GeoDataFrame)):
        df = _df.copy()
    else:
        df = pd.DataFrame(_df)

    need = ["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    df = df[need].copy()
    df["vacancy_pct"] = pd.to_numeric(df["vacancy_pct"], errors="coerce")
    df["avg_travel_min"] = pd.to_numeric(df["avg_travel_min"], errors="coerce")
    df = df.dropna(subset=["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min"])
    if df.empty:
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
    df["v_n"] = norm01_clipped(df["vacancy_pct"].values, v_lo, v_hi)
    df["t_n"] = norm01_clipped(df["avg_travel_min"].values, t_lo, t_hi)

    rng = np.random.default_rng(seed)
    used = set()
    rows = []
    tries = 0
    while len(rows) < n and tries < 8000:
        tries += 1
        A = df.sample(1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
        if A["GEMEINDE_NAME"] in used:
            continue

        orientation = int(rng.integers(0, 2))
        if orientation == 0:
            cand = df[
                (df["v_n"] < A["v_n"])
                & (df["t_n"] < A["t_n"])
                & ((A["v_n"] - df["v_n"]).between(0.10, 0.30, inclusive="both"))
                & ((A["t_n"] - df["t_n"]).between(0.05, 0.18, inclusive="both"))
                & (~df["GEMEINDE_NAME"].isin(used | {A["GEMEINDE_NAME"]}))
            ]
        else:
            cand = df[
                (df["v_n"] > A["v_n"])
                & (df["t_n"] > A["t_n"])
                & ((df["v_n"] - A["v_n"]).between(0.10, 0.30, inclusive="both"))
                & ((df["t_n"] - A["t_n"]).between(0.05, 0.18, inclusive="both"))
                & (~df["GEMEINDE_NAME"].isin(used | {A["GEMEINDE_NAME"]}))
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

# ---------- Preference elicitation ----------
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

# Track answers (optional quick-fit kept minimal)
choices_dict, answered_rows = {}, []
for _, r in sc.iterrows():
    key = f"choice_{int(r.qid)}"
    ch = st.session_state.get(key, None)
    if ch in ("A", "B"):
        choices_dict[key] = ch
        answered_rows.append(r)
ans_sig = answers_signature(choices_dict) if choices_dict else None

if answered_rows and st.session_state.get("answers_sig") != ans_sig:
    st.session_state["manual_weights"] = {
        "w0": 0.0,
        "w_v": 2.0,
        "w_t": 2.0,
        "v_min": float(v_range[0]),
        "v_max": float(v_range[1]),
        "t_min": float(t_range[0]),
        "t_max": float(t_range[1]),
    }
    st.session_state["answers_sig"] = ans_sig

# ---------- Data preview ----------
with tab_data:
    show = df.copy()
    show["avg_travel_min_eff"] = pd.to_numeric(show["avg_travel_min_eff"], errors="coerce")
    cols = ["GEMEINDE_NAME", "vacancy_pct", "avg_travel_min_eff", "preference_score"]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols].head(30).rename(columns={"avg_travel_min_eff": "avg_travel_min"}))
    st.caption(f"Total rows: {len(show)}")

# ---------- Map ----------
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

        if "manual_weights" in st.session_state:
            w0 = float(st.session_state["manual_weights"].get("w0", 0.0))
            a = float(st.session_state["manual_weights"].get("w_v", 2.0))
            b = float(st.session_state["manual_weights"].get("w_t", 2.0))
            v_lo, v_hi = float(st.session_state["manual_weights"].get("v_min", v_range[0])), float(
                st.session_state["manual_weights"].get("v_max", v_range[1])
            )
            t_lo, t_hi = float(st.session_state["manual_weights"].get("t_min", t_range[0])), float(
                st.session_state["manual_weights"].get("t_max", t_range[1])
            )
        else:
            w0, a, b = resolve_prior(meta, st.session_state)
            v_lo, v_hi = v_range
            t_lo, t_hi = t_range

        v_all = norm01_clipped(pd.to_numeric(df["vacancy_pct"], errors="coerce").values, v_lo, v_hi)
        t_all = norm01_clipped(
            pd.to_numeric(df["avg_travel_min_eff"], errors="coerce").values, t_lo, t_hi
        )
        pen_t = commute_penalty_shape(t_all, penalty_k)
        util = w0 + a * v_all - b * pen_t

        finite = np.isfinite(util)
        if finite.any():
            util = util - np.median(util[finite])

        df_render = df.copy()
        df_render["preference_score"] = 100.0 * (1.0 / (1.0 + np.exp(-util)))
        vmin, vmax = 0.0, 100.0
        df_render = df_render.dropna(subset=["preference_score"])

    # Folium renderer
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
