# -*- coding: utf-8 -*-
"""
UKTA Feedback - Offline policy builder (Excel-optimized + tqdm + teacher-score parsing)

기능 요약
- 입력: XLSX/CSV/TSV/Parquet (XLSX는 헤더만 1pass로 읽고 필요한 컬럼만 usecols로 재로딩)
- 우수 집단(Top q%) 기준 타깃 밴드(P_low–P_high) 계산 + 방향(up/down/neutral)
- 분리력/일관성/안정성: Cohen's d, AUROC, Spearman, bootstrap(부호·순위 안정성)
- 고상관(|Spearman ρ| ≥ rho_thresh) 패밀리(연결요소)로 중복 제거
- 패밀리 대표 점수화 → 앵커(6~10개) 선정
- 진행상황 tqdm (—no_progress로 끄기)
- 교사 점수 파싱: 'essay_score_avg'처럼 '#' 연결 문자열을 teacher_* 컬럼들로 변환
  - --teacher_names 로 순서명 직접 지정 (예: grammar,vocabulary,...)
  - --teacher_autolink 로 예측 점수(essay_score_*)와 스피어만 상관으로 자동 정렬/리네이밍

사용 예(루브릭 하나):
  python ukta_policy_builder_tqdm.py \
    --data /path/feat_train_500.xlsx \
    --score_col teacher_grammar \
    --groupby essay_level \
    --q_top 0.80 --p_low 40 --p_high 70 \
    --n_boot 100 --topk_rank 40 --rho_thresh 0.8 \
    --outdir /path/artifacts_teacher/grammar \
    --downcast --no_progress
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# ---------------------------
# Meta / Score columns to exclude from features
# ---------------------------
META_COLS_HARD = {
    "essay", "essay_id", "essay_len", "essay_level", "essay_type", "essay_prompt",
    "essay_main_subject", "morpheme_language", "morpheme_sentences",
    "correction_language", "correction_origin", "correction_revised", "correction_revisedSentences"
}
SCORE_COL_PREFIXES = ("essay_score",)   # 예측 점수/총점 접두
SCORE_COL_EXACT = {
    "essay_score_a","essay_score_b","essay_score_c",
    "essay_score_avg","essay_score_T","essay_scoreT_avg"
}
TEACHER_COL_PREFIXES = ("teacher_",)    # 교사 점수 접두(피처에서 제외)

# ---------------------------
# Small helpers
# ---------------------------
def _tqdm(iterable, desc="", disable=False, total=None):
    return tqdm(iterable, desc=desc, disable=disable, total=total)

def downcast_numeric(df: pd.DataFrame):
    """Optional: memory saver. Keep floats as float32, ints as int32."""
    for c in df.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
    return df

# ---------------------------
# I/O (Excel optimized)
# ---------------------------
def load_table(path: Path, score_col: str, groupby_cols=None, extra_cols=None, progress=False):
    """
    XLSX: 1) header-only pass -> 2) decide columns -> 3) read usecols only
    Other formats: direct read.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    elif ext in [".xlsx", ".xls"]:
        if not path.exists():
            raise FileNotFoundError(path)
        header_df = pd.read_excel(path, nrows=0, engine="openpyxl")
        cols_all = header_df.columns.tolist()

        keep = set()
        meta_hard = set(META_COLS_HARD)
        score_prefixes = SCORE_COL_PREFIXES
        score_exact = set(SCORE_COL_EXACT)

        # 필수: score_col + groupby + extra_cols(teacher_source 등)
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

        # 후보: 점수/메타가 아닌 나머지(자질 후보)
        for c in cols_all:
            if c in meta_hard or c in score_exact:
                continue
            if any(c.startswith(p) for p in score_prefixes):
                continue
            keep.add(c)

        usecols_final = sorted(list(keep))
        if progress:
            print(f"📄 XLSX selective loading: using {len(usecols_final)}/{len(cols_all)} columns")

        df = pd.read_excel(path, engine="openpyxl", usecols=usecols_final)
        return df
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def infer_feature_cols(df: pd.DataFrame, extra_drop_prefixes=()):
    """
    - Take all numeric columns
    - Drop known score/meta columns
    """
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
# Teacher score parsing
# ---------------------------
def attach_teacher_scores(
    df: pd.DataFrame,
    source_col: str = "essay_score_avg",
    delim: str = "#",
    rubric_names: list | None = None,
    auto_link_pred: bool = True,
    verbose: bool = True
):
    """
    '2.6#2.6#2.0#...' 문자열을 파싱해 teacher_* 컬럼을 생성/부착.
    - rubric_names 제공 시: teacher_{name} 로 컬럼명 지정(순서 그대로)
    - auto_link_pred=True 인 경우:
        예측 점수(essay_score_*)와 Spearman|ρ| 최대 매칭으로 teacher_*를 리네이밍
        (suffix를 재사용: essay_score_grammar -> teacher_grammar)
    """
    if source_col not in df.columns:
        if verbose:
            print(f"ℹ️ teacher source '{source_col}' 없음: 파싱 생략")
        return df, []

    s = df[source_col].astype(str).fillna("")
    toks = s.str.split(delim)
    maxlen = toks.map(len).max()

    n = len(df)
    arr = np.full((n, maxlen), np.nan, dtype=np.float32)
    for i, t in enumerate(toks):
        L = min(len(t), maxlen)
        for j in range(L):
            try:
                arr[i, j] = float(t[j])
            except:
                arr[i, j] = np.nan

    # 임시 이름
    if rubric_names and len(rubric_names) == maxlen:
        names = [f"teacher_{name.strip()}" for name in rubric_names]
    else:
        names = [f"teacher_rubric_{j}" for j in range(maxlen)]

    for j, name in enumerate(names):
        df[name] = arr[:, j]

    # 자동 정렬/리네이밍
    created_cols = list(names)
    if auto_link_pred:
        pred_cols = [c for c in df.columns if c.startswith("essay_score_") and c not in SCORE_COL_EXACT]
        if len(pred_cols) >= 1 and len(created_cols) >= 1:
            # |Spearman| 상관행렬
            R = np.zeros((len(created_cols), len(pred_cols)), dtype=float)
            for j, tcol in enumerate(created_cols):
                for k, pcol in enumerate(pred_cols):
                    mask = df[[tcol, pcol]].notna().all(axis=1).values
                    if mask.sum() >= 3:
                        rho, _ = stats.spearmanr(df.loc[mask, tcol], df.loc[mask, pcol])
                        R[j, k] = abs(rho) if np.isfinite(rho) else 0.0
            used_t = set(); used_p = set(); pairs = []
            while len(pairs) < min(len(created_cols), len(pred_cols)):
                best = (-1, -1, -1.0)
                for j in range(len(created_cols)):
                    if j in used_t: continue
                    for k in range(len(pred_cols)):
                        if k in used_p: continue
                        if R[j, k] > best[2]:
                            best = (j, k, R[j, k])
                j, k, score = best
                if j == -1: break
                used_t.add(j); used_p.add(k); pairs.append((j, k, score))
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
# Helper stats
# ---------------------------
def cohens_d(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
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
    x = np.asarray(feat); y = np.asarray(is_top).astype(int)
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
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return rho

# ---------------------------
# Core policy builders
# ---------------------------
def compute_topq_mask(df, score_col, groupby=None, q=0.8):
    if groupby:
        def mark(g):
            thr = np.nanpercentile(g[score_col].values, q*100)
            return g[score_col] >= thr
        return df.groupby(groupby, dropna=False, group_keys=False).apply(mark)
    else:
        thr = np.nanpercentile(df[score_col].values, q*100)
        return df[score_col] >= thr

def compute_target_bands(df, feat_cols, is_top, groupby=None, p_low=40, p_high=70, progress=False):
    rows = []
    if groupby:
        grouped_all = df.groupby(groupby, dropna=False)
        grouped_top = df.loc[is_top].groupby(groupby, dropna=False)
        groups = list(grouped_all.groups.keys())
        groups = [g if isinstance(g, tuple) else (g,) for g in groups]
        for g in _tqdm(groups, desc="Target bands (groups)", disable=not progress):
            all_idx = grouped_all.groups[g if len(groupby)>1 else g[0]]
            top_idx = grouped_top.groups.get(g if len(groupby)>1 else g[0], [])
            for f in feat_cols:
                all_vals = df.loc[all_idx, f].dropna().values
                top_vals = df.loc[top_idx, f].dropna().values if len(top_idx)>0 else np.array([])
                if top_vals.size < 10:
                    band_low, band_high, direction = np.nan, np.nan, "neutral"
                else:
                    band_low = float(np.nanpercentile(top_vals, p_low))
                    band_high = float(np.nanpercentile(top_vals, p_high))
                    med_top = float(np.nanmedian(top_vals))
                    med_all = float(np.nanmedian(all_vals)) if all_vals.size else np.nan
                    if np.isnan(med_all):
                        direction = "neutral"
                    else:
                        direction = "up" if med_top>med_all else ("down" if med_top<med_all else "neutral")
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
                    direction = "up" if med_top>med_all else ("down" if med_top<med_all else "neutral")
            rows.append({
                "feature_id": f, "cohort": "ALL",
                "band_low": band_low, "band_high": band_high, "direction": direction
            })
    return pd.DataFrame(rows)

def discriminative_stats(df, feat_cols, score_col, is_top, n_boot=100, topk_rank=40, seed=42, progress=False):
    y_cont = df[score_col].values.astype(float)
    y_bin  = is_top.values.astype(int)

    base_auc = {}
    base_d   = {}
    base_rho = {}
    for f in _tqdm(feat_cols, desc="Base stats (d/AUC/Spearman)", disable=not progress, total=len(feat_cols)):
        x = df[f].values.astype(float)
        base_auc[f] = auc_pos_vs_neg(x, y_bin)
        base_d[f]   = abs(cohens_d(x[y_bin==1], x[y_bin==0]))
        base_rho[f] = spearman_corr(x, y_cont)

    auc_series = pd.Series(base_auc)
    fill_val = (np.nanmin(auc_series.values) - 1.0) if not np.isnan(auc_series.values).all() else -1.0
    auc_filled = auc_series.fillna(fill_val)
    base_rank = auc_filled.rank(method="average", ascending=False)
    in_topk = set(base_rank[base_rank <= topk_rank].index)

    rng = np.random.default_rng(seed)
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
            xb = df[f].values[idx].astype(float)
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

def correlation_families(df, feat_cols, rho_thresh=0.8, progress=False):
    if progress:
        print("🔗 Computing Spearman |rho| matrix...")
    X = df[feat_cols].copy()
    rho = X.corr(method="spearman").abs()

    unvisited = set(feat_cols)
    families = []
    pbar = _tqdm(total=len(feat_cols), desc="Families (components)", disable=not progress)
    while unvisited:
        start = unvisited.pop()
        stack = [start]
        comp = {start}
        while stack:
            u = stack.pop()
            neigh = set(rho.index[(rho.loc[u] >= rho_thresh).values])
            neigh = {v for v in neigh if v in unvisited}
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

def default_highlightable(feature_id: str) -> bool:
    prefixes = ("adjacency_", "similarity_", "sentenceLvl_", "readability_")
    return any(feature_id.startswith(p) for p in prefixes)

def score_family_reps(stats_df, families_df, highlight_bonus=1.2):
    df = stats_df.merge(families_df, on="feature_id", how="left")
    for col in ["auc","d_abs","boot_sign_agree","boot_rank_stability"]:
        mu = df[col].mean(skipna=True)
        sd = df[col].std(ddof=1, skipna=True)
        if not np.isfinite(sd) or sd == 0:
            z = (df[col] - mu)
        else:
            z = (df[col] - mu) / sd
        df[f"z_{col}"] = z.replace([np.inf, -np.inf], 0).fillna(0)
    df["score"] = df[["z_auc","z_d_abs","z_boot_sign_agree","z_boot_rank_stability"]].sum(axis=1)
    df["score"] *= df["feature_id"].map(lambda f: highlight_bonus if default_highlightable(f) else 1.0)
    return df

def select_candidates(stats_df,
                      d_min=0.25, auc_min=0.58, rho_min=0.20,
                      boot_sign_min=0.70, boot_rank_min=0.70):
    s = stats_df.copy()
    cond = (
        ((s["d_abs"] >= d_min) | (s["auc"] >= auc_min)) &
        (s["spearman"].abs() >= rho_min) &
        (s["boot_sign_agree"] >= boot_sign_min) &
        (s["boot_rank_stability"] >= boot_rank_min)
    )
    s["is_candidate"] = cond
    return s

def choose_anchors(scored_rep_df, min_k=6, max_k=10):
    rep_sorted = scored_rep_df.sort_values("score", ascending=False)
    selected = rep_sorted.head(max_k).copy()
    if len(selected) < min_k:
        pass
    selected["rationale"] = "high discriminative power + stable; easy to explain (check manually)"
    selected["highlightable"] = selected["feature_id"].map(lambda f: default_highlightable(f))
    return selected[["feature_id","score","rationale","highlightable","family_id"]]

# ---------------------------
# Main (CLI)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to table (XLSX/CSV/TSV/Parquet)")
    ap.add_argument("--score_col", default="essay_scoreT_avg", help="Column used to define top-q group")
    ap.add_argument("--groupby", default="", help="Comma-separated cohort keys (e.g., essay_level). Empty -> ALL")
    ap.add_argument("--q_top", type=float, default=0.80, help="Top quantile for 'excellent' group (0.8 = top20%)")
    ap.add_argument("--p_low", type=int, default=40, help="Low percentile for target band")
    ap.add_argument("--p_high", type=int, default=70, help="High percentile for target band")
    ap.add_argument("--n_boot", type=int, default=100, help="Bootstrap runs")
    ap.add_argument("--topk_rank", type=int, default=40, help="Top-K for rank stability")
    ap.add_argument("--rho_thresh", type=float, default=0.8, help="Correlation threshold for family grouping")
    ap.add_argument("--d_min", type=float, default=0.25, help="Candidate filter: |d| minimum")
    ap.add_argument("--auc_min", type=float, default=0.58, help="Candidate filter: AUC minimum")
    ap.add_argument("--rho_min", type=float, default=0.20, help="Candidate filter: |Spearman| minimum")
    ap.add_argument("--boot_sign_min", type=float, default=0.70, help="Candidate filter: sign agreement minimum")
    ap.add_argument("--boot_rank_min", type=float, default=0.70, help="Candidate filter: rank stability minimum")
    ap.add_argument("--min_anchors", type=int, default=6)
    ap.add_argument("--max_anchors", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    ap.add_argument("--downcast", action="store_true", help="Downcast numeric dtypes (float32/int32) to save memory")
    # teacher parsing options
    ap.add_argument("--parse_teacher", action="store_true", default=True,
                    help="Parse teacher scores from a composite column into teacher_* columns (default: True)")
    ap.add_argument("--teacher_source", default="essay_score_avg",
                    help="Column containing '#' joined teacher scores")
    ap.add_argument("--teacher_delim", default="#", help="Delimiter for teacher scores")
    ap.add_argument("--teacher_names", default="",
                    help="Comma-separated rubric names to assign (e.g., 'grammar,vocabulary,...'). If blank, auto names.")
    ap.add_argument("--teacher_autolink", action="store_true", default=True,
                    help="Auto align teacher_* to essay_score_* by Spearman correlation and rename")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    path = Path(args.data)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress

    # teacher parsing needs teacher_source column present when reading XLSX
    extra_cols = []
    if args.parse_teacher and args.teacher_source:
        extra_cols.append(args.teacher_source)

    # 0) Load (Excel optimized selective loading)
    groupby = [c.strip() for c in args.groupby.split(",") if c.strip()]
    df = load_table(path, score_col=args.score_col,
                    groupby_cols=groupby if groupby else None,
                    extra_cols=extra_cols, progress=show_progress)

    if args.downcast:
        df = downcast_numeric(df)

    # 0-1) Attach teacher scores
    if args.parse_teacher:
        rubric_names = [s.strip() for s in args.teacher_names.split(",") if s.strip()] or None
        df, created_cols = attach_teacher_scores(
            df,
            source_col=args.teacher_source,
            delim=args.teacher_delim,
            rubric_names=rubric_names,
            auto_link_pred=args.teacher_autolink,
            verbose=show_progress
        )
        if show_progress and created_cols:
            print("📎 Available teacher columns:", ", ".join(created_cols))

    if args.score_col not in df.columns:
        raise ValueError(f"'{args.score_col}' column not found in data. "
                         f"현재 컬럼들 예: {', '.join(list(df.columns)[:12])} ...")

    # 1) Feature columns
    feat_cols = infer_feature_cols(df)
    if not feat_cols:
        raise ValueError("No feature columns found. Check numeric columns and exclude rules.")
    if show_progress:
        print(f"🔢 Features detected: {len(feat_cols)}")

    # 2) Top-q mask (excellent set)
    is_top = compute_topq_mask(df, args.score_col, groupby=groupby if groupby else None, q=args.q_top)
    if show_progress:
        thr = np.nanpercentile(df[args.score_col].values, args.q_top*100)
        print(f"🏁 Top-{int((1-args.q_top)*100)}% cutoff @ {thr:.4f} (col: {args.score_col})")

    # 3) Target bands
    bands_df = compute_target_bands(df, feat_cols, is_top, groupby=groupby if groupby else None,
                                    p_low=args.p_low, p_high=args.p_high, progress=show_progress)
    bands_df.to_csv(outdir / "target_bands.csv", index=False, encoding="utf-8-sig")

    # 4) Discriminative / consistency / stability stats
    stats_df = discriminative_stats(df, feat_cols, args.score_col, is_top,
                                    n_boot=args.n_boot, topk_rank=args.topk_rank,
                                    seed=args.seed, progress=show_progress)
    cand_df = select_candidates(stats_df,
                                d_min=args.d_min, auc_min=args.auc_min, rho_min=args.rho_min,
                                boot_sign_min=args.boot_sign_min, boot_rank_min=args.boot_rank_min)
    stats_out = cand_df.copy()
    stats_out.to_csv(outdir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 5) Families on candidates only
    cand_feats = stats_out.loc[stats_out["is_candidate"], "feature_id"].tolist()
    if len(cand_feats) == 0:
        fallback = stats_out.sort_values(["auc","d_abs"], ascending=False).head(60)["feature_id"].tolist()
        cand_feats = fallback
        if show_progress:
            print(f"⚠️ No candidates passed thresholds; using fallback top-{len(cand_feats)} by AUC/|d|")

    fam_df = correlation_families(df, cand_feats, rho_thresh=args.rho_thresh, progress=show_progress)
    fam_df.to_csv(outdir / "families.csv", index=False, encoding="utf-8-sig")

    # 6) Choose family representatives & anchors
    rep_input = stats_out[stats_out["feature_id"].isin(cand_feats)]
    rep_score_df = score_family_reps(rep_input, fam_df)
    anchors = choose_anchors(rep_score_df, min_k=args.min_anchors, max_k=args.max_anchors)
    anchors.insert(0, "rubric", args.score_col)  # rubric key = score_col
    anchors.to_csv(outdir / "anchors.csv", index=False, encoding="utf-8-sig")

    # 7) Summary
    thr = np.nanpercentile(df[args.score_col].values, args.q_top*100)
    summary = pd.DataFrame([{
        "n_rows": len(df),
        "n_features_total": len(feat_cols),
        "n_candidates": int(stats_out["is_candidate"].sum()),
        "score_q": args.q_top,
        "score_q_threshold": thr,
        "n_topq": int(is_top.sum()),
        "pct_topq": float(is_top.mean()*100.0),
        "groupby": ",".join(groupby) if groupby else "ALL",
        "p_low": args.p_low,
        "p_high": args.p_high,
        "rho_thresh": args.rho_thresh
    }])
    summary.to_csv(outdir / "topq_split_summary.csv", index=False, encoding="utf-8-sig")

    print("✅ Done.")
    print(f"- Target bands:        {outdir / 'target_bands.csv'}")
    print(f"- Feature importance:  {outdir / 'feature_importance.csv'}")
    print(f"- Families:            {outdir / 'families.csv'}")
    print(f"- Anchors:             {outdir / 'anchors.csv'}")
    print(f"- Summary:             {outdir / 'topq_split_summary.csv'}")

if __name__ == "__main__":
    main()
