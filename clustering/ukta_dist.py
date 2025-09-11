# -*- coding: utf-8 -*-
"""
UKTA Feedback - Data-driven Policy Builder (prefix-free, rubric-specific)

핵심 특징
- 상위집단 정의 선택: (A) 분위수(quantile) 또는 (B) 정확히 특정 점수(eq, 예: 3.0)
- 교란 제거: 길이/문장수/문단길이 + 다른 루브릭 점수들을 통제한 '부분 스피어만 상관' 지원
- 안정 선택: L1 로지스틱 + 서브샘플 안정성(stability selection)으로 진짜 필요한 피처만 선별
- 특이성: 문법(타깃 루브릭)에 더 특이적인 피처를 밀어주고, 타 루브릭에 공통인 신호는 감점
- 이름(접두어) 없이 순수 데이터 기반으로 "루브릭 핵심 자질"을 선정
- 산출물: target_bands.csv / feature_importance.csv / families.csv / anchors.csv / topq_split_summary.csv

사용 예(문법 루브릭, 3점=우수, 데이터 기반 선택 전부 활성화):
python ukta_policy_builder_data_driven.py \
  --data /path/feat_train_500.xlsx \
  --score_col teacher_grammar \
  --top_select eq --top_value 3.0 \
  --groupby essay_level,essay_type \
  --selector_modes partial_spearman,l1logit,specificity \
  --partial_controls auto \
  --l1_subsamples 60 --l1_sample_frac 0.7 --l1_C 0.7 --l1_select_pmin 0.6 \
  --rho_weight 1.8 --specificity_weight 1.0 \
  --n_boot 150 --topk_rank 30 \
  --outdir artifacts_teacher/grammar_data_driven
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

warnings.simplefilter("ignore", ConstantInputWarning)

# ---------------------------
# Meta / Score columns to exclude from feature pool
# ---------------------------
META_COLS_HARD = {
    "essay", "essay_id", "essay_len", "essay_level", "essay_type", "essay_prompt",
    "essay_main_subject", "morpheme_language", "morpheme_sentences",
    "correction_language", "correction_origin", "correction_revised", "correction_revisedSentences"
}
SCORE_COL_PREFIXES = ("essay_score",)
SCORE_COL_EXACT = {
    "essay_score_a", "essay_score_b", "essay_score_c",
    "essay_score_avg", "essay_score_T", "essay_scoreT_avg"
}
TEACHER_COL_PREFIXES = ("teacher_",)

# ---------------------------
# Small helpers
# ---------------------------
def _tqdm(iterable=None, desc="", disable=False, total=None):
    if iterable is None:
        return tqdm(total=total, desc=desc, disable=disable)
    return tqdm(iterable, desc=desc, disable=disable, total=total)

def downcast_numeric(df: pd.DataFrame):
    for c in df.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
    return df

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(skipna=True), s.std(ddof=1, skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return (s - mu).fillna(0)
    return ((s - mu) / sd).replace([np.inf, -np.inf], 0).fillna(0)

# ---------------------------
# I/O (Excel optimized)
# ---------------------------
def load_table(path: Path, score_col: str, groupby_cols=None, extra_cols=None, progress=False):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        return pd.read_parquet(path)

    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)

    if ext in [".xlsx", ".xls"]:
        if not path.exists():
            raise FileNotFoundError(path)
        header_df = pd.read_excel(path, nrows=0, engine="openpyxl")
        cols_all = header_df.columns.tolist()

        keep = set()
        meta_hard = set(META_COLS_HARD)
        score_exact = set(SCORE_COL_EXACT)

        if score_col in cols_all:
            keep.add(score_col)
        if groupby_cols:
            for g in groupby_cols:
                if g in cols_all:
                    keep.add(g)
        if extra_cols:
            for c in extra_cols:
                if c in cols_all:
                    keep.add(c)

        for c in cols_all:
            if c in meta_hard or c in score_exact:
                continue
            if any(c.startswith(p) for p in SCORE_COL_PREFIXES):
                continue
            keep.add(c)

        usecols_final = sorted(list(keep))
        if progress:
            print(f"📄 XLSX selective loading: using {len(usecols_final)}/{len(cols_all)} columns")

        df = pd.read_excel(path, engine="openpyxl", usecols=usecols_final)
        return df

    raise ValueError(f"Unsupported file type: {ext}")

def infer_feature_cols(df: pd.DataFrame, extra_drop_prefixes=()):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_exact = set(SCORE_COL_EXACT) | set(df.columns.intersection(META_COLS_HARD))
    drop_prefixes = SCORE_COL_PREFIXES + TEACHER_COL_PREFIXES + tuple(extra_drop_prefixes)

    feat_cols = []
    for c in num_cols:
        if c in drop_exact:
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            continue
        feat_cols.append(c)
    return feat_cols

# ---------------------------
# Teacher score parser (optional)
# ---------------------------
def attach_teacher_scores(
    df: pd.DataFrame,
    source_col: str = "essay_score_avg",
    delim: str = "#",
    rubric_names: list | None = None,
    auto_link_pred: bool = True,
    verbose: bool = True
):
    if source_col not in df.columns:
        if verbose:
            print(f"ℹ️ teacher source '{source_col}' 없음: 파싱 생략")
        return df, []

    s = df[source_col].astype(str).fillna("")
    toks = s.str.split(delim)
    maxlen = toks.map(len).max() if len(toks) else 0
    if maxlen == 0:
        if verbose:
            print("ℹ️ teacher source 비어있음: 파싱 생략")
        return df, []

    n = len(df)
    arr = np.full((n, maxlen), np.nan, dtype=np.float32)
    for i, t in enumerate(toks):
        L = min(len(t), maxlen)
        for j in range(L):
            try:
                val = t[j].strip()
                arr[i, j] = float(val) if val != "" else np.nan
            except Exception:
                arr[i, j] = np.nan

    if rubric_names and len(rubric_names) == maxlen:
        names = [f"teacher_{name.strip()}" for name in rubric_names]
    else:
        names = [f"teacher_rubric_{j}" for j in range(maxlen)]

    for j, name in enumerate(names):
        df[name] = arr[:, j]

    created_cols = list(names)

    if auto_link_pred:
        pred_cols = [c for c in df.columns if c.startswith("essay_score_") and c not in SCORE_COL_EXACT]
        if len(pred_cols) >= 1 and len(created_cols) >= 1:
            R = np.zeros((len(created_cols), len(pred_cols)), dtype=float)
            for j, tcol in enumerate(created_cols):
                for k, pcol in enumerate(pred_cols):
                    sub = df[[tcol, pcol]].dropna()
                    if len(sub) >= 3:
                        rho, _ = stats.spearmanr(sub[tcol].values, sub[pcol].values)
                        R[j, k] = abs(rho) if np.isfinite(rho) else 0.0
                    else:
                        R[j, k] = 0.0
            used_t = set(); used_p = set(); pairs = []
            while len(pairs) < min(len(created_cols), len(pred_cols)):
                j_best, k_best, best = -1, -1, -1.0
                for j in range(len(created_cols)):
                    if j in used_t:
                        continue
                    for k in range(len(pred_cols)):
                        if k in used_p:
                            continue
                        if R[j, k] > best:
                            j_best, k_best, best = j, k, R[j, k]
                if j_best == -1:
                    break
                used_t.add(j_best); used_p.add(k_best); pairs.append((j_best, k_best, best))
            rename_map = {}
            for j, k, _ in pairs:
                suffix = pred_cols[k].replace("essay_score_", "")
                rename_map[created_cols[j]] = f"teacher_{suffix}"
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
                created_cols = [rename_map.get(c, c) for c in created_cols]

    if verbose:
        print("🧩 attached teacher columns:", ", ".join(created_cols))
    return df, created_cols

# ---------------------------
# Metrics
# ---------------------------
def cohens_d(pos, neg):
    pos = np.asarray(pos, dtype=float); neg = np.asarray(neg, dtype=float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    m1, m0 = pos.mean(), neg.mean()
    s1, s0 = pos.std(ddof=1), neg.std(ddof=1)
    n1, n0 = len(pos), len(neg)
    denom = (n1+n0-2)
    if denom <= 0:
        return np.nan
    sp = math.sqrt(((n1-1)*s1*s1 + (n0-1)*s0*s0) / denom)
    if sp == 0 or np.isnan(sp):
        return np.nan
    d = (m1 - m0) / sp
    J = 1 - (3/(4*(n1+n0)-9)) if (n1+n0)>9 else 1.0
    return d * J

def auc_pos_vs_neg(feat, is_top):
    x = np.asarray(feat, dtype=float)
    y = np.asarray(is_top).astype(int)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() == 0:
        return np.nan
    y_mask = y[mask]
    if y_mask.sum() == 0 or (mask.sum() - y_mask.sum()) == 0:
        return np.nan
    try:
        return roc_auc_score(y_mask, x[mask])
    except ValueError:
        return np.nan

def spearman_corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return rho if np.isfinite(rho) else np.nan

# ---------------------------
# Core: top-q vs eq=3.0
# ---------------------------
def compute_topq_mask_quantile(df, score_col, groupby=None, q=0.8):
    if groupby:
        quant = df.groupby(groupby, dropna=False)[score_col].transform(lambda s: np.nanpercentile(s, q*100))
        return df[score_col] >= quant
    thr = np.nanpercentile(df[score_col].values, q*100)
    return df[score_col] >= thr

# ---------------------------
# Partial Spearman (controls)
# ---------------------------
def partial_spearman(x: pd.Series, y: pd.Series, Z: pd.DataFrame) -> float:
    from numpy.linalg import lstsq
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    if Z is None or Z.shape[1] == 0:
        return spearman_corr(x.values, y.values)
    Z = Z.copy().astype(float).fillna(0.0)
    mask = ~x.isna() & ~y.isna()
    if mask.sum() < (Z.shape[1] + 3):
        return np.nan
    XZ = np.c_[np.ones(mask.sum()), Z.loc[mask].values]
    try:
        beta_x, *_ = lstsq(XZ, x.loc[mask].values, rcond=None)
        rx = x.loc[mask].values - XZ.dot(beta_x)
        beta_y, *_ = lstsq(XZ, y.loc[mask].values, rcond=None)
        ry = y.loc[mask].values - XZ.dot(beta_y)
        rho, _ = stats.spearmanr(rx, ry)
        return float(rho) if np.isfinite(rho) else np.nan
    except Exception:
        return np.nan

# ---------------------------
# L1-logistic stability selection
# ---------------------------
def l1_logit_stability(X_df: pd.DataFrame, y_series: pd.Series,
                       C=1.0, subsamples=50, sample_frac=0.7, random_state=42,
                       standardize=True) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise RuntimeError("scikit-learn이 필요합니다. `pip install scikit-learn` 후 다시 실행하세요.") from e

    rng = np.random.default_rng(random_state)
    X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    if standardize:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        X = (X - mu) / sd
    y = y_series.values.astype(int)

    n, m = X.shape
    counts = np.zeros(m, dtype=int)
    for _ in range(subsamples):
        idx = rng.choice(n, int(max(10, round(n*sample_frac))), replace=False)
        clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=2000)
        clf.fit(X[idx], y[idx])
        sel = (np.abs(clf.coef_[0]) > 1e-10)
        counts += sel.astype(int)
    return counts / subsamples

# ---------------------------
# Target bands
# ---------------------------
def compute_target_bands(df, feat_cols, is_top, groupby=None, p_low=40, p_high=70, progress=False):
    if not (0 <= p_low <= p_high <= 100):
        raise ValueError("0 <= p_low <= p_high <= 100 를 만족해야 합니다.")
    rows = []
    if groupby:
        grouped_all = df.groupby(groupby, dropna=False)
        grouped_top = df.loc[is_top].groupby(groupby, dropna=False)
        groups = list(grouped_all.groups.keys())
        groups = [g if isinstance(g, tuple) else (g,) for g in groups]
        for g in _tqdm(groups, desc="Target bands (groups)", disable=not progress):
            key = g if len(groupby) > 1 else g[0]
            all_idx = grouped_all.groups[key]
            top_idx = grouped_top.groups.get(key, [])
            for f in feat_cols:
                av = df.loc[all_idx, f].dropna().values
                tv = df.loc[top_idx, f].dropna().values if len(top_idx)>0 else np.array([])
                if tv.size < 10:
                    band_low, band_high, direction = np.nan, np.nan, "neutral"
                else:
                    band_low = float(np.nanpercentile(tv, p_low))
                    band_high = float(np.nanpercentile(tv, p_high))
                    med_top = float(np.nanmedian(tv))
                    med_all = float(np.nanmedian(av)) if av.size else np.nan
                    if np.isnan(med_all):
                        direction = "neutral"
                    else:
                        direction = "up" if med_top > med_all else ("down" if med_top < med_all else "neutral")
                rows.append({
                    "feature_id": f,
                    "cohort": "|".join(map(str, g)),
                    "band_low": band_low,
                    "band_high": band_high,
                    "direction": direction
                })
    else:
        top = df.loc[is_top, feat_cols]
        all_vals = df[feat_cols]
        for f in _tqdm(feat_cols, desc="Target bands (ALL)", disable=not progress, total=len(feat_cols)):
            tv = top[f].dropna().values
            av = all_vals[f].dropna().values
            if tv.size < 10:
                band_low, band_high, direction = np.nan, np.nan, "neutral"
            else:
                band_low = float(np.nanpercentile(tv, p_low))
                band_high = float(np.nanpercentile(tv, p_high))
                med_top = float(np.nanmedian(tv))
                med_all = float(np.nanmedian(av)) if av.size else np.nan
                if np.isnan(med_all):
                    direction = "neutral"
                else:
                    direction = "up" if med_top > med_all else ("down" if med_top < med_all else "neutral")
            rows.append({
                "feature_id": f, "cohort": "ALL",
                "band_low": band_low, "band_high": band_high, "direction": direction
            })
    return pd.DataFrame(rows)

# ---------------------------
# Discriminative / stability stats (baseline)
# ---------------------------
def discriminative_stats(df, feat_cols, score_col, is_top, n_boot=100, topk_rank=40, seed=42, progress=False):
    rng = np.random.default_rng(seed)
    y_cont = df[score_col].astype(float).values
    y_bin  = is_top.astype(int).values

    base_auc, base_d, base_rho = {}, {}, {}
    for f in _tqdm(feat_cols, desc="Base stats (d/AUC/Spearman)", disable=not progress, total=len(feat_cols)):
        x = df[f].astype(float).values
        base_auc[f] = auc_pos_vs_neg(x, y_bin)
        base_d[f]   = abs(cohens_d(x[y_bin==1], x[y_bin==0]))
        base_rho[f] = spearman_corr(x, y_cont)

    auc_series = pd.Series(base_auc)
    fill_val = (np.nanmin(auc_series.values) - 1.0) if not np.isnan(auc_series.values).all() else -1.0
    auc_filled = auc_series.fillna(fill_val)
    base_rank = auc_filled.rank(method="average", ascending=False)
    in_topk = set(base_rank[base_rank <= topk_rank].index)

    sign_agree = {f:0 for f in feat_cols}
    rank_hits  = {f:0 for f in feat_cols}
    base_sign  = {f: np.sign(base_rho[f]) if not np.isnan(base_rho[f]) else 0 for f in feat_cols}

    n = len(df)
    for _ in _tqdm(range(n_boot), desc=f"Bootstrap x{n_boot}", disable=not progress, total=n_boot):
        idx = rng.integers(0, n, size=n)
        yb_cont = y_cont[idx]; yb_bin = y_bin[idx]
        auc_b = {}
        rho_b_sign = {}
        for f in feat_cols:
            xb = df[f].astype(float).values[idx]
            auc_b[f] = auc_pos_vs_neg(xb, yb_bin)
            rho_b = spearman_corr(xb, yb_cont)
            rho_b_sign[f] = np.sign(rho_b) if not np.isnan(rho_b) else 0
        auc_b_series = pd.Series(auc_b)
        fill_b = (np.nanmin(auc_b_series.values) - 1.0) if not np.isnan(auc_b_series.values).all() else -1.0
        rank_b = auc_b_series.fillna(fill_b).rank(method="average", ascending=False)
        topk_b = set(rank_b[rank_b <= topk_rank].index)
        for f in feat_cols:
            if base_sign[f] != 0 and rho_b_sign[f] == base_sign[f]:
                sign_agree[f] += 1
            if f in in_topk and f in topk_b:
                rank_hits[f] += 1

    rows = []
    for f in feat_cols:
        rows.append({
            "feature_id": f,
            "d_abs": base_d[f],
            "auc": base_auc[f],
            "spearman": base_rho[f],
            "boot_sign_agree": sign_agree[f] / n_boot,
            "boot_rank_stability": rank_hits[f] / n_boot
        })
    return pd.DataFrame(rows)

# ---------------------------
# Correlation families
# ---------------------------
def correlation_families(df, feat_cols, rho_thresh=0.8, progress=False):
    if progress:
        print("🔗 Computing Spearman |rho| matrix...")
    X = df[feat_cols].copy()
    rho = X.corr(method="spearman").abs().fillna(0.0)
    for c in rho.columns:
        rho.loc[c, c] = 0.0

    unvisited = set(feat_cols)
    families = []
    pbar = _tqdm(None, desc="Families (components)", disable=not progress, total=len(feat_cols))
    while unvisited:
        start = unvisited.pop()
        stack = [start]
        comp = {start}
        while stack:
            u = stack.pop()
            neigh = set(rho.index[(rho.loc[u] >= rho_thresh).values])
            neigh = {v for v in neigh if v in unvisited}
            if neigh:
                comp |= neigh
                stack.extend(list(neigh))
                unvisited -= neigh
        families.append(sorted(list(comp)))
        pbar.update(len(comp))
    pbar.close()

    mapping = {}
    for i, fam in enumerate(families):
        for f in fam:
            mapping[f] = i
    fam_df = pd.DataFrame([{"feature_id": f, "family_id": mapping[f]} for f in feat_cols])
    return fam_df

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to table (XLSX/CSV/TSV/Parquet)")
    ap.add_argument("--score_col", default="essay_scoreT_avg", help="Target rubric column (e.g., teacher_grammar)")
    ap.add_argument("--groupby", default="", help="Comma-separated cohort keys (e.g., essay_level,essay_type). Empty -> ALL")

    # top selection
    ap.add_argument("--top_select", choices=["quantile", "eq"], default="quantile",
                    help="상위집단 정의 방식: 분위수(quantile) 또는 정확히 같은 값(eq)")
    ap.add_argument("--top_value", type=float, default=3.0, help="--top_select eq 일 때 상위로 볼 점수값")
    ap.add_argument("--q_top", type=float, default=0.80, help="Top quantile if --top_select=quantile (0.8=top20%)")

    # data-driven selectors
    ap.add_argument("--selector_modes", default="partial_spearman,l1logit,specificity",
                    help="콤마로 조합: partial_spearman, l1logit, specificity")
    ap.add_argument("--partial_controls", default="auto",
                    help="부분상관 통제 컬럼. 'auto'면 모든 teacher_*(타깃 제외)+규모변수 자동 선택")

    ap.add_argument("--l1_subsamples", type=int, default=50)
    ap.add_argument("--l1_sample_frac", type=float, default=0.7)
    ap.add_argument("--l1_C", type=float, default=1.0)
    ap.add_argument("--l1_select_pmin", type=float, default=0.6, help="안정선택 최소 선택확률")

    ap.add_argument("--rho_weight", type=float, default=1.5, help="|partial rho| 가중치")
    ap.add_argument("--specificity_weight", type=float, default=1.0, help="특이성 가중치")
    ap.add_argument("--partial_rho_min", type=float, default=0.15, help="후보 필터: |partial rho| 하한")
    ap.add_argument("--specificity_min", type=float, default=0.0, help="후보 필터: 특이성 하한")

    # baseline stats / families / anchors
    ap.add_argument("--n_boot", type=int, default=100, help="Bootstrap runs")
    ap.add_argument("--topk_rank", type=int, default=40, help="Top-K for rank stability")
    ap.add_argument("--rho_thresh", type=float, default=0.8, help="Correlation threshold for family grouping")
    ap.add_argument("--min_anchors", type=int, default=6)
    ap.add_argument("--max_anchors", type=int, default=10)

    # output & perf
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--downcast", action="store_true")
    ap.add_argument("--outdir", default="artifacts")

    # teacher parsing (optional)
    group_teacher = ap.add_argument_group("teacher")
    group_teacher.add_argument("--parse_teacher", dest="parse_teacher", action="store_true", help="Parse teacher scores")
    group_teacher.add_argument("--no_parse_teacher", dest="parse_teacher", action="store_false", help="Disable teacher parsing")
    group_teacher.set_defaults(parse_teacher=True)
    group_teacher.add_argument("--teacher_source", default="essay_score_avg", help="Column with '#' joined teacher scores")
    group_teacher.add_argument("--teacher_delim", default="#")
    group_teacher.add_argument("--teacher_names", default="")
    group_teacher.add_argument("--teacher_autolink", action="store_true", default=True)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # 0) load
    path = Path(args.data)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    groupby = [c.strip() for c in args.groupby.split(",") if c.strip()]
    show_progress = not args.no_progress

    extra_cols = []
    if args.parse_teacher and args.teacher_source:
        extra_cols.append(args.teacher_source)

    df = load_table(path, score_col=args.score_col,
                    groupby_cols=groupby if groupby else None,
                    extra_cols=extra_cols, progress=show_progress)

    if args.downcast:
        df = downcast_numeric(df)

    # teacher parse
    created_cols = []
    if args.parse_teacher:
        rubric_names = [s.strip() for s in args.teacher_names.split(",") if s.strip()] or None
        df, created_cols = attach_teacher_scores(
            df, source_col=args.teacher_source, delim=args.teacher_delim,
            rubric_names=rubric_names, auto_link_pred=args.teacher_autolink, verbose=show_progress
        )
        if show_progress and created_cols:
            print("📎 teacher columns:", ", ".join(created_cols))
        if created_cols:
            pd.DataFrame({"teacher_columns": created_cols}).to_csv(outdir / "teacher_columns.csv", index=False, encoding="utf-8-sig")

    if args.score_col not in df.columns:
        raise ValueError(f"'{args.score_col}' not found. 현재 컬럼 예시: {', '.join(list(df.columns)[:15])} ...")

    # 1) feature pool
    feat_cols = infer_feature_cols(df)
    if not feat_cols:
        raise ValueError("No numeric feature columns found after exclusion rules.")
    if show_progress:
        print(f"🔢 Features detected: {len(feat_cols)}")

    # 2) top/equal mask
    if args.top_select == "eq":
        is_top = pd.Series(np.isclose(df[args.score_col].astype(float), args.top_value), index=df.index)
    else:
        if not (0.0 < args.q_top < 1.0):
            raise ValueError("--q_top must be in (0,1).")
        is_top = compute_topq_mask_quantile(df, args.score_col, groupby=groupby if groupby else None, q=args.q_top)

    if show_progress:
        if args.top_select == "eq":
            print(f"🏁 Top=EQ({args.top_value}) → pct_top={is_top.mean()*100:.2f}%")
        else:
            thr = np.nanpercentile(df[args.score_col].values, args.q_top*100)
            print(f"🏁 Top-{int((1-args.q_top)*100)}% cutoff @ {thr:.4f} (col: {args.score_col}) | pct_top={is_top.mean()*100:.2f}%")

    # 3) target bands (우수 분포 기반 목표)
    bands_df = compute_target_bands(df, feat_cols, is_top, groupby=groupby if groupby else None,
                                    p_low=40, p_high=70, progress=show_progress)
    bands_df.to_csv(outdir / "target_bands.csv", index=False, encoding="utf-8-sig")

    # 4) baseline discriminative stats
    stats_df = discriminative_stats(df, feat_cols, args.score_col, is_top,
                                    n_boot=args.n_boot, topk_rank=args.topk_rank,
                                    seed=args.seed, progress=show_progress)

    # 5) controls(Z) for partial correlation
    if args.partial_controls == "auto":
        teacher_cols = [c for c in df.columns if c.startswith("teacher_")]
        teacher_cols = [c for c in teacher_cols if c != args.score_col]
        # 흔한 규모/길이 후보 중 존재하는 것만 채택
        size_candidates = [
            "basic_count_word_Cnt", "basic_count_sent_Cnt",
            "sentenceLvl_char_paraLenAvg", "sentenceLvl_word_paraLenAvg",
            "essay_len"
        ]
        basic_controls = [c for c in size_candidates if c in df.columns]
        Z_cols = list(dict.fromkeys(teacher_cols + basic_controls))
    else:
        Z_cols = [c.strip() for c in args.partial_controls.split(",") if c.strip()]
    Z = df[Z_cols].copy() if Z_cols else pd.DataFrame(index=df.index)

    # 6) data-driven signals
    modes = set([s.strip() for s in args.selector_modes.split(",") if s.strip()])

    # (a) partial spearman |rho|
    partial_rhos = {}
    if "partial_spearman" in modes:
        y_cont = df[args.score_col].astype(float)
        for f in _tqdm(feat_cols, desc="Partial Spearman(|rho|)", disable=not show_progress, total=len(feat_cols)):
            r = partial_spearman(df[f], y_cont, Z)
            partial_rhos[f] = abs(r) if np.isfinite(r) else np.nan
    else:
        partial_rhos = {f: np.nan for f in feat_cols}

    # (b) L1-logit stability selection (eq/quantile 모두 y_bin으로 가능)
    l1_probs = {}
    if "l1logit" in modes:
        X_df = df[feat_cols].copy()
        y_bin = is_top.astype(int)
        l1_prob_arr = l1_logit_stability(
            X_df, y_bin, C=args.l1_C,
            subsamples=args.l1_subsamples, sample_frac=args.l1_sample_frac,
            random_state=args.seed, standardize=True
        )
        l1_probs = {f: float(p) for f, p in zip(feat_cols, l1_prob_arr)}
    else:
        l1_probs = {f: np.nan for f in feat_cols}

    # (c) specificity: grammar vs other teachers
    specificity_scores = {}
    if "specificity" in modes:
        other_teachers = [c for c in df.columns if c.startswith("teacher_") and c != args.score_col]
        y_cont = df[args.score_col].astype(float)
        for f in _tqdm(feat_cols, desc="Specificity(Grammar vs Others)", disable=not show_progress, total=len(feat_cols)):
            rho_g = partial_spearman(df[f], y_cont, Z) if "partial_spearman" in modes else spearman_corr(df[f].values, y_cont.values)
            rho_g = abs(rho_g) if np.isfinite(rho_g) else np.nan
            rho_others = []
            for tcol in other_teachers:
                ry = partial_spearman(df[f], df[tcol].astype(float), Z) if "partial_spearman" in modes else spearman_corr(df[f].values, df[tcol].astype(float).values)
                rho_others.append(abs(ry) if np.isfinite(ry) else 0.0)
            max_other = max(rho_others) if len(rho_others) > 0 else 0.0
            specificity_scores[f] = (rho_g - max_other) if np.isfinite(rho_g) else np.nan
    else:
        specificity_scores = {f: np.nan for f in feat_cols}

    # 7) merge & score
    stats_df["partial_rho_abs"] = stats_df["feature_id"].map(partial_rhos)
    stats_df["l1_select_prob"] = stats_df["feature_id"].map(l1_probs)
    stats_df["specificity"]    = stats_df["feature_id"].map(specificity_scores)

    stats_df["score"] = (
        zscore(stats_df["auc"].fillna(stats_df["auc"].median())) +
        zscore(stats_df["d_abs"].fillna(stats_df["d_abs"].median())) +
        args.rho_weight * zscore(stats_df["partial_rho_abs"].fillna(0)) +
        zscore(stats_df["boot_sign_agree"].fillna(0)) +
        zscore(stats_df["boot_rank_stability"].fillna(0)) +
        args.specificity_weight * zscore(stats_df["specificity"].fillna(0))
    )

    # 8) candidate filter (prefix-free)
    conds = []
    # partial rho gate
    if "partial_spearman" in modes:
        conds.append((stats_df["partial_rho_abs"].fillna(0) >= args.partial_rho_min))
    # l1 stability gate
    if "l1logit" in modes:
        conds.append((stats_df["l1_select_prob"].fillna(0) >= args.l1_select_pmin))
    # specificity gate
    if "specificity" in modes:
        conds.append((stats_df["specificity"].fillna(0) >= args.specificity_min))
    if len(conds) == 0:
        is_cand = pd.Series(True, index=stats_df.index)
    else:
        is_cand = conds[0]
        for c in conds[1:]:
            is_cand = is_cand & c
    stats_df["is_candidate"] = is_cand

    stats_out = stats_df.copy()
    stats_out.to_csv(outdir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 9) families on candidates (fallback if none)
    cand_feats = stats_out.loc[stats_out["is_candidate"], "feature_id"].tolist()
    if len(cand_feats) == 0:
        fallback = stats_out.sort_values(["score","auc","d_abs"], ascending=False).head(60)["feature_id"].tolist()
        cand_feats = fallback
        if show_progress:
            print(f"⚠️ No candidates passed thresholds; using fallback top-{len(cand_feats)} by score/AUC/|d|")

    fam_df = correlation_families(df, cand_feats, rho_thresh=args.rho_thresh, progress=show_progress)
    fam_df.to_csv(outdir / "families.csv", index=False, encoding="utf-8-sig")

    # 10) choose anchors (family representatives by score)
    rep_input = stats_out[stats_out["feature_id"].isin(cand_feats)]
    rep_score_df = rep_input.merge(fam_df, on="feature_id", how="left")
    rep_sorted = rep_score_df.sort_values("score", ascending=False)
    anchors = rep_sorted.groupby("family_id", dropna=False, as_index=False).head(1)
    # 패밀리 대표 상위에서 min~max 범위 확보
    anchors = anchors.sort_values("score", ascending=False).head(args.max_anchors)
    if len(anchors) < args.min_anchors:
        # 부족하면 점수순으로 보충
        needed = args.min_anchors - len(anchors)
        extra = rep_sorted[~rep_sorted["feature_id"].isin(anchors["feature_id"])].head(needed)
        anchors = pd.concat([anchors, extra], ignore_index=True)

    anchors = anchors[["feature_id","score","family_id","auc","d_abs","partial_rho_abs","l1_select_prob","specificity","boot_sign_agree","boot_rank_stability"]]
    anchors.insert(0, "rubric", args.score_col)
    anchors["rationale"] = "data-driven: discrimination + partial corr + stability + specificity"
    anchors.to_csv(outdir / "anchors.csv", index=False, encoding="utf-8-sig")

    # 11) summary
    if args.top_select == "quantile":
        thr = np.nanpercentile(df[args.score_col].values, args.q_top*100)
    else:
        thr = float(args.top_value)

    summary = pd.DataFrame([{
        "n_rows": int(len(df)),
        "n_features_total": int(len(feat_cols)),
        "n_candidates": int(stats_out["is_candidate"].sum()),
        "selector_modes": ",".join(sorted(list(modes))),
        "score_col": args.score_col,
        "top_select": args.top_select,
        "top_value_or_q": args.top_value if args.top_select == "eq" else args.q_top,
        "score_threshold": thr,
        "n_top": int(is_top.sum()),
        "pct_top": float(is_top.mean()*100.0),
        "groupby": ",".join(groupby) if groupby else "ALL",
        "rho_thresh": args.rho_thresh,
        "rho_weight": args.rho_weight,
        "specificity_weight": args.specificity_weight,
        "partial_rho_min": args.partial_rho_min,
        "specificity_min": args.specificity_min,
        "l1_subsamples": args.l1_subsamples,
        "l1_sample_frac": args.l1_sample_frac,
        "l1_C": args.l1_C,
        "l1_select_pmin": args.l1_select_pmin,
        "seed": args.seed
    }])
    summary.to_csv(outdir / "topq_split_summary.csv", index=False, encoding="utf-8-sig")

    print("✅ Done.")
    print(f"- Target bands:        {outdir / 'target_bands.csv'}")
    print(f"- Feature importance:  {outdir / 'feature_importance.csv'}")
    print(f"- Families:            {outdir / 'families.csv'}")
    print(f"- Anchors:             {outdir / 'anchors.csv'}")
    print(f"- Summary:             {outdir / 'topq_split_summary.csv'}")
    if created_cols:
        print(f"- Teacher columns:     {outdir / 'teacher_columns.csv'}")

if __name__ == "__main__":
    main()
