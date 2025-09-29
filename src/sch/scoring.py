import numpy as np
import pandas as pd

def robust_range(series, qlo=1, qhi=99):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty: return (0.0, 1.0)
    lo = float(np.nanpercentile(s, qlo))
    hi = float(np.nanpercentile(s, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi

def norm01(x, lo, hi):
    x = np.asarray(x, dtype="float64")
    lo, hi = float(lo), float(hi)
    if hi <= lo + 1e-9: return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def exp_penalty_monotone(t_norm, k):
    t = np.clip(np.asarray(t_norm, dtype="float64"), 0.0, 1.0)
    k = float(k)
    if k <= 1e-6: return t
    return (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k))

def blend_preference(vacancy, commute, a=2.0, b=2.0, k=3.0):
    v_min, v_max = robust_range(vacancy)
    t_min, t_max = robust_range(commute)
    v_all = norm01(pd.to_numeric(vacancy, errors="coerce").values, v_min, v_max)
    t_all = norm01(pd.to_numeric(commute, errors="coerce").values, t_min, t_max)
    pen_t = exp_penalty_monotone(t_all, k)
    util0 = a * v_all - b * pen_t
    w0 = -np.median(util0[np.isfinite(util0)])
    score = 100.0 * (1.0 / (1.0 + np.exp(-(w0 + util0))))
    return score, dict(w0=float(w0), a=float(a), b=float(b), k=float(k),
                       v_min=float(v_min), v_max=float(v_max),
                       t_min=float(t_min), t_max=float(t_max))
