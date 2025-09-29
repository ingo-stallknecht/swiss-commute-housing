import numpy as np
import pandas as pd
from sch.scoring import robust_range, norm01, exp_penalty_monotone, blend_preference

def test_robust_range_basic():
    lo, hi = robust_range(pd.Series([0, 1, 2, 3, 100]))
    assert lo <= 1 and hi >= 3

def test_norm01_identity_on_range():
    x = np.array([0.0, 5.0, 10.0])
    y = norm01(x, 0.0, 10.0)
    assert np.allclose(y, [0.0, 0.5, 1.0])

def test_exp_penalty_monotone_bounds():
    t = np.linspace(0, 1, 5)
    y = exp_penalty_monotone(t, k=3.0)
    assert 0.0 <= y.min() <= y.max() <= 1.0

def test_blend_preference_shapes():
    vacancy = pd.Series([0.01, 0.03, 0.02, 0.0, np.nan])
    commute = pd.Series([10, 30, 50, 70, 20])
    score, meta = blend_preference(vacancy, commute)
    assert len(score) == len(vacancy)
    assert all(k in meta for k in ["a", "b", "k", "v_min", "v_max", "t_min", "t_max"])
