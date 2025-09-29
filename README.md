# 🏠🚆 Swiss Housing & Commute Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
[![Tests](https://github.com/ingo-stallknecht/swiss-commute-housing/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ingo-stallknecht/swiss-commute-housing/actions/workflows/tests.yml)
[![CI](https://github.com/ingo-stallknecht/swiss-commute-housing/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ingo-stallknecht/swiss-commute-housing/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://swiss-commute-housing-ivg9a6hhq3j5gkaq9yintl.streamlit.app/)
![Last Commit](https://img.shields.io/github/last-commit/ingo-stallknecht/swiss-commute-housing)

Interactive **Streamlit app** to explore the trade-off between **housing availability** and **public transport accessibility** across Swiss municipalities.

- **Data**: housing vacancy shares + GTFS rail timetables
- **Geospatial**: municipality polygons, centroid heuristics, LV95/CH ↔ WGS84
- **Artifacts**: reproducible builder exports GeoJSON/Parquet/CSV for the app
- **App**: interactive map + controls for housing/commute weights and penalty curvature

---

## 🚀 Live Demo

👉 **Streamlit App:** [Live link here](https://swiss-commute-housing-ivg9a6hhq3j5gkaq9yintl.streamlit.app/)

👉 **Colab Notebook:** [Notebook link here](https://colab.research.google.com/github/ingo-stallknecht/swiss-commute-housing/blob/main/notebooks/swiss_commute_housing.ipynb)

📸 *Screenshot of the app interface:*
![App Screenshot](assets/screenshot_app.png)

---

## 🧩 Problem Statement

Housing decisions balance **availability** and **accessibility**.
This tool helps answer:

> “Which Swiss municipalities combine **higher housing vacancy** with **shorter commute times** from my chosen SBB station?”

---

## ⚙️ Technical Approach

This project demonstrates a full **data → geospatial → ML-style scoring → app deployment** pipeline:

### 1. Data Sources
- **Housing vacancy**: BFS *Leerwohnungszählung* (annual survey of vacant dwellings).
  - BFS exports come in *multi-header CSV format*.
  - Implemented a **robust parser** (`io_utils.py`) that auto-detects header lines and normalizes percentage values.
- **GTFS transport**: Official SBB GTFS feed (`stops.txt`, `trips.txt`, `routes.txt`, `stop_times.txt`).
- **Geospatial data**: Gemeinde boundaries from OpenDataSoft.
  - CRS conversions: LV95 (EPSG:2056) ↔ WGS84 (EPSG:4326).
  - Geometry simplification for efficient app rendering.

### 2. Commute Time Computation
- Built a **station graph** from GTFS tables (`gtfs_graph.py`).
- Implemented **shortest-path traversal** to compute **average travel minutes** from any chosen origin station to all municipalities.
- Aggregated station-level travel times to **municipality polygons** by centroid mapping.

### 3. Preference Scoring
- Normalize vacancy (%) and commute minutes into [0,1] (robust quantile range).
- Apply an **exponential penalty** to commute time (long commutes get penalized more).
- Combine housing + commute via a **logistic utility** → 0–100 preference score.
- The app exposes **commute penalty curvature `k`** as a user control for sensitivity to travel time.

### 4. Reproducible Artifacts
- Automated pipeline (`make_artifacts.py`) builds:
  - `gemeinden.geojson` (full geometry)
  - `gemeinden_simplified.geojson` (lightweight polygons)
  - `gemeinden_centroids.parquet` (map hover points)
  - `meta.json` (scoring parameters)
  - `tt_by_origin.parquet` (multi-origin commute times)

### 5. Deployment
- **Streamlit UI** (`app/app.py`) with Folium/Deck.gl maps.
- **Dockerized + GitHub CI** for reproducibility.
- Hosted on **Streamlit Cloud** with automatic redeploys on push.

---

## ✨ Key Features

- **Interactive dashboard**:
  - Switch between *preference score*, *housing only*, *commute only*
  - Choose different SBB origins dynamically

📸 *Example views (insert before publishing):*
- Housing only heatmap → `assets/map_housing_only.png`
- Commute only heatmap → `assets/map_commute_only.png`
- Preference score combined → `assets/screenshot_app.png`

---

## 🛠️ How to Run Locally

Clone, install, build artifacts, and run the app:

```bash
git clone https://github.com/ingo-stallknecht/swiss-commute-housing.git
cd swiss-commute-housing

# Create and activate virtualenv
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .

# Place input data in data/
#   data/vacancy_municipality.csv
#   data/gtfs_train.zip

# Build artifacts
.venv/Scripts/python scripts/make_artifacts.py

# Run the app
.venv/Scripts/python scripts/launch_app.py
