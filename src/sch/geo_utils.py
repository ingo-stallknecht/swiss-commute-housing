import re
import unicodedata as ucn

import geopandas as gpd
import numpy as np

CRS_WGS84 = "EPSG:4326"
CRS_CH = "EPSG:2056"


def norm_name(s: str) -> str:
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
