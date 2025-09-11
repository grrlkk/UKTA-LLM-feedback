# ukta_builder/feature_selectors.py
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats
from .utils import _tqdm
from .stats_calculators import spearman_corr

def partial_spearman(x: pd.Series, y: pd.Series, Z: pd.DataFrame) -> float:
    """
    x, y 사이의 부분 스피어만 순위상관.
    Z(다른 teacher_* 점수들)를 선형회귀로 제거한 잔차끼리 spearmanr.
    """
    x, y = pd.Series(x).astype(float), pd.Series(y).astype(float)
    if Z is None or getattr(Z, "shape", (0, 0))[1] == 0:
        return spearman_corr(x.values, y.values)

    Z = Z.copy().astype(float).fillna(0.0)
    mask = ~x.isna() & ~y.isna()
    n_eff = int(mask.sum())
    if n_eff < (Z.shape[1] + 3):
        return np.nan

    XZ = np.c_[np.ones(n_eff), Z.loc[mask].values]
    try:
        beta_x, *_ = lstsq(XZ, x.loc[mask].values, rcond=None)
        rx = x.loc[mask].values - XZ.dot(beta_x)
        beta_y, *_ = lstsq(XZ, y.loc[mask].values, rcond=None)
        ry = y.loc[mask].values - XZ.dot(beta_y)
        rho, _ = stats.spearmanr(rx, ry)
        return float(rho) if np.isfinite(rho) else np.nan
    except Exception:
        return np.nan

def l1_logit_stability(X_df, y_series, C, subsamples, sample_frac, seed, standardize=True):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise RuntimeError("scikit-learn is required. Please install it via `pip install scikit-learn`.")
    rng = np.random.default_rng(seed)
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    if standardize:
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        X = (X - mu) / sd
    y = y_series.values.astype(int)

    n, m = X.shape
    counts = np.zeros(m, dtype=int)
    for _ in range(subsamples):
        idx_n = int(max(10, round(n * sample_frac)))
        if idx_n >= n: idx_n = n - 1
        idx = rng.choice(n, idx_n, replace=False)
        if np.unique(y[idx]).size < 2:
            continue
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=C, max_iter=2000,
            random_state=int(rng.integers(1e9)), class_weight="balanced"
        )
        clf.fit(X[idx], y[idx])
        counts += (np.abs(clf.coef_[0]) > 1e-10).astype(int)
    return counts / subsamples if subsamples > 0 else np.zeros(m)

def correlation_families(df, feat_cols, rho_thresh, progress=False):
    if not feat_cols:
        return pd.DataFrame(columns=['feature_id', 'family_id'])
    X = df[feat_cols].copy()
    rho = X.corr(method="spearman").abs()

    unvisited, families = set(feat_cols), []
    while unvisited:
        start = unvisited.pop()
        comp, stack = {start}, [start]
        while stack:
            u = stack.pop()
            neighbors = set(rho.index[rho[u] >= rho_thresh]) & unvisited
            comp.update(neighbors)
            stack.extend(list(neighbors))
            unvisited.difference_update(neighbors)
        families.append(sorted(list(comp)))
    mapping = {f: i for i, fam in enumerate(families) for f in fam}
    return pd.DataFrame([{"feature_id": f, "family_id": mapping[f]} for f in feat_cols])
