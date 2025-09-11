# ukta_builder/stats_calculators.py
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from .utils import _tqdm

def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2: return np.nan
    m1, m0 = pos.mean(), neg.mean()
    s1, s0 = pos.std(ddof=1), neg.std(ddof=1)
    n1, n0 = len(pos), len(neg)
    denom = (n1 + n0 - 2)
    if denom <= 0: return np.nan
    sp = math.sqrt(((n1 - 1) * s1 * s1 + (n0 - 1) * s0 * s0) / denom)
    if sp == 0 or np.isnan(sp): return np.nan
    return (m1 - m0) / sp

def auc_pos_vs_neg(feat, is_top):
    x, y = np.asarray(feat, dtype=float), np.asarray(is_top).astype(int)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() == 0 or np.unique(y[mask]).size < 2: return np.nan
    try:
        return roc_auc_score(y[mask], x[mask])
    except ValueError:
        return np.nan

def spearman_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3: return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return rho if np.isfinite(rho) else np.nan

def compute_target_bands(df, feat_cols, is_top, p_low=40, p_high=70, progress=False):
    rows = []
    top_df = df.loc[is_top]
    for f in _tqdm(feat_cols, desc="Target bands", disable=not progress):
        tv = top_df[f].dropna().values
        av = df[f].dropna().values
        if tv.size < 10:
            band_low, band_high, direction = np.nan, np.nan, "neutral"
        else:
            band_low = float(np.nanpercentile(tv, p_low))
            band_high = float(np.nanpercentile(tv, p_high))
            med_top = float(np.nanmedian(tv))
            med_all = float(np.nanmedian(av)) if av.size > 0 else np.nan
            direction = "neutral" if np.isnan(med_all) else ("up" if med_top > med_all else "down")
        rows.append({"feature_id": f, "cohort": "ALL", "band_low": band_low, "band_high": band_high, "direction": direction})
    return pd.DataFrame(rows)

def discriminative_stats(df, feat_cols, score_col, is_top, n_boot, topk_rank, seed, progress=False):
    rng = np.random.default_rng(seed)
    y_cont, y_bin = df[score_col].astype(float).values, is_top.astype(int).values
    
    base_metrics = {}
    for f in _tqdm(feat_cols, desc="Base stats", disable=not progress):
        x = df[f].astype(float).values
        base_metrics[f] = {
            "d_abs": abs(cohens_d(x[y_bin == 1], x[y_bin == 0])),
            "auc": auc_pos_vs_neg(x, y_bin),
            "spearman": spearman_corr(x, y_cont)
        }
    
    auc_series = pd.Series({f: v['auc'] for f, v in base_metrics.items()})
    base_rank = auc_series.fillna(auc_series.min() - 1).rank(method="average", ascending=False)
    in_topk = set(base_rank[base_rank <= topk_rank].index)

    boot_counts = {f: {'sign_agree': 0, 'rank_hits': 0} for f in feat_cols}
    base_signs = {f: np.sign(v['spearman']) if pd.notna(v['spearman']) else 0 for f, v in base_metrics.items()}

    n = len(df)
    for _ in _tqdm(range(n_boot), desc=f"Bootstrap x{n_boot}", disable=not progress):
        idx = rng.integers(0, n, size=n)
        yb_cont, yb_bin = y_cont[idx], y_bin[idx]
        
        boot_auc = {}
        for f in feat_cols:
            xb = df[f].astype(float).values[idx]
            rho_b = spearman_corr(xb, yb_cont)
            if base_signs[f] != 0 and np.sign(rho_b) == base_signs[f]:
                boot_counts[f]['sign_agree'] += 1
            boot_auc[f] = auc_pos_vs_neg(xb, yb_bin)
        
        boot_auc_series = pd.Series(boot_auc)
        rank_b = boot_auc_series.fillna(boot_auc_series.min() - 1).rank(method="average", ascending=False)
        topk_b = set(rank_b[rank_b <= topk_rank].index)
        
        for f in in_topk:
            if f in topk_b:
                boot_counts[f]['rank_hits'] += 1
                
    rows = []
    for f in feat_cols:
        rows.append({
            "feature_id": f,
            "d_abs": base_metrics[f]["d_abs"],
            "auc": base_metrics[f]["auc"],
            "spearman": base_metrics[f]["spearman"],
            "boot_sign_agree": boot_counts[f]['sign_agree'] / n_boot if n_boot > 0 else 0,
            "boot_rank_stability": boot_counts[f]['rank_hits'] / n_boot if n_boot > 0 and f in in_topk else 0
        })
    return pd.DataFrame(rows)

def quick_screen(df, feat_cols, score_col, is_top):
    """
    값싼 지표(AUC, |rho|, |d|)만으로 빠르게 랭킹을 만드는 초경량 스크린.
    combo = z(AUC) + z(|rho|) + z(|d_abs|).
    """
    y_cont = df[score_col].astype(float).values
    y_bin = is_top.astype(int).values
    rows = []
    for f in feat_cols:
        x = df[f].astype(float).values
        rows.append({
            "feature_id": f,
            "d_abs": abs(cohens_d(x[y_bin==1], x[y_bin==0])),
            "auc": auc_pos_vs_neg(x, y_bin),
            "spearman": spearman_corr(x, y_cont),
        })
    q = pd.DataFrame(rows)
    from .utils import zscore
    q["combo"] = (
        zscore(q["auc"].fillna(0.5))
        + zscore(q["spearman"].abs().fillna(0))
        + zscore(q["d_abs"].fillna(0))
    )
    return q
