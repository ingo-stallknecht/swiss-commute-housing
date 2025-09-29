import io, re, json
import pandas as pd
import numpy as np

def read_bfs_like_csv(path: str) -> pd.DataFrame:
    text = None
    for enc in ("utf-8-sig","utf-8","latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except Exception:
            pass
    if text is None:
        raise IOError(f"Cannot read file {path} with utf-8/latin1.")
    lines = text.splitlines()
    hdr_idx, pat = 0, re.compile(r"(TIME_PERIOD|Zeitperiode).*(OBS_VALUE|Beobachtungswert)")
    for i, ln in enumerate(lines[:200]):
        if pat.search(ln): hdr_idx = i; break
    return pd.read_csv(io.StringIO("\n".join(lines[hdr_idx:])), sep=";", engine="python", dtype=str)

def parse_percent(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    if not s: return np.nan
    s = s.replace("\xa0"," ").replace("%","").strip()
    s = s.replace("’","").replace("'","").replace(" ", "").replace(",", ".")
    try:
        v = float(s)
    except:
        return np.nan
    return v/100.0 if v > 10 else v
