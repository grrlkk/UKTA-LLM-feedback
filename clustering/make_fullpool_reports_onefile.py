
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

# ---------- Utils ----------
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(ddof=1, skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return (s - mu).fillna(0)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], 0).fillna(0)

def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2: return np.nan
    m1, m0 = pos.mean(), neg.mean()
    s1, s0 = pos.std(ddof=1), neg.std(ddof=1)
    n1, n0 = len(pos), len(neg)
    denom = (n1 + n0 - 2)
    if denom <= 0: return np.nan
    sp = np.sqrt(((n1 - 1)*s1*s1 + (n0 - 1)*s0*s0) / denom)
    if sp == 0 or np.isnan(sp): return np.nan
    return (m1 - m0) / sp

def auc_pos_vs_neg(x, y_bin):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_bin, dtype=int)
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

def partial_spearman(x: pd.Series, y: pd.Series, Z: pd.DataFrame) -> float:
    x, y = pd.Series(x).astype(float), pd.Series(y).astype(float)
    if Z is None or getattr(Z, "shape", (0, 0))[1] == 0:
        return spearman_corr(x.values, y.values)
    try:
        from sklearn.linear_model import Ridge
    except Exception:
        # Fallback: un-partial Spearman
        return spearman_corr(x.values, y.values)
    Z = Z.copy().astype(float).fillna(0.0)
    mask = ~x.isna() & ~y.isna()
    if mask.sum() < (Z.shape[1] + 3): return np.nan
    rr = Ridge(alpha=1e-3, fit_intercept=True)
    rr.fit(Z.loc[mask].values, x.loc[mask].values)
    rx = x.loc[mask].values - rr.predict(Z.loc[mask].values)
    rr.fit(Z.loc[mask].values, y.loc[mask].values)
    ry = y.loc[mask].values - rr.predict(Z.loc[mask].values)
    rho, _ = stats.spearmanr(rx, ry)
    return float(rho) if np.isfinite(rho) else np.nan

# ---------- I/O ----------
def load_table(path: Path):
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {ext}")

def attach_teacher_scores(df: pd.DataFrame, source_col: str, delim: str, rubric_names: list | None):
    if source_col not in df.columns:
        return df, []
    s = df[source_col].astype(str).fillna("")
    toks = s.str.split(delim)
    maxlen = toks.map(len).max()
    if maxlen == 0: return df, []
    n = len(df)
    arr = np.full((n, maxlen), np.nan, dtype=np.float32)
    for i, t in enumerate(toks):
        if not isinstance(t, list): t = []
        L = min(len(t), maxlen)
        for j in range(L):
            try:
                val = str(t[j]).strip()
                arr[i, j] = float(val) if val != "" else np.nan
            except (ValueError, TypeError):
                arr[i, j] = np.nan
    if rubric_names and len(rubric_names) == maxlen:
        names = [f"teacher_{name.strip()}" for name in rubric_names]
    else:
        names = [f"teacher_rubric_{j}" for j in range(maxlen)]
    for j, name in enumerate(names):
        df[name] = arr[:, j]
    return df, names

META_COLS_HARD = {
    "essay","essay_id","essay_len","essay_level","essay_type","essay_prompt",
    "essay_main_subject","morpheme_language","morpheme_sentences",
    "correction_language","correction_origin","correction_revised","correction_revisedSentences"
}
SCORE_COL_EXACT = {
    "essay_score_a","essay_score_b","essay_score_c",
    "essay_score_avg","essay_score_T","essay_scoreT_avg"
}
SCORE_COL_PREFIXES = ("essay_score",)
TEACHER_COL_PREFIXES = ("teacher_",)

def infer_feature_cols(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_exact = set(SCORE_COL_EXACT).union(set(df.columns.intersection(META_COLS_HARD)))
    drop_prefixes = SCORE_COL_PREFIXES + TEACHER_COL_PREFIXES
    feat_cols = [c for c in num_cols if c not in drop_exact and not any(c.startswith(p) for p in drop_prefixes)]
    return feat_cols

# ---------- Meta tagging ----------
def _extract_pos_token(name: str) -> str | None:
    s = name.strip()
    m = re.search(r"^NDW_([^_]+)_", s) or re.search(r"^ttr_([^_]+)_", s, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r"^basic_(count|density|list)_([^_]+)_(Cnt|Den|Lst)$", s, re.IGNORECASE)
    if m: return m.group(2)
    m = re.search(r"overlap_(noun|verb|adjective|adverb|function|content|all|lemma)", s, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r"^sentenceLvl_([A-Za-z]+)_", s)
    if m: return m.group(1)
    return None

def _pos_token_to_bucket(tok: str | None) -> str:
    if tok is None: return "OTHER"
    t = tok.strip().upper()
    if t in {"NN","NNG","NNP","NNB","NP","NR","N","NNC","NNPC","NNBC","NMC","NM","NC","NOUN"}: return "NOUN"
    if t in {"VV","VX","V","VC","VERB"}: return "VERB"
    if t in {"VA","VCN","VCP","ADJ","ADJECTIVE"}: return "ADJ"
    if t in {"MA","MAG","MAJ","ADV","ADVERB"}: return "ADV"
    if t in {"MM","M","MMC","DET"}: return "DET"
    if t in {"J","E","X","F","IC","FUNCTION","FUNC"}: return "FUNC"
    if t.startswith("V"): return "VERB" if t in {"VV","VX","V"} else ("ADJ" if t in {"VA","VCN","VCP"} else "VERB")
    if t.startswith("N"): return "NOUN"
    if t.startswith("M"): return "DET"
    return "OTHER"

def _match_regex_first(name: str, buckets: list[dict], default: str="OtherType") -> str:
    s = name.strip()
    for b in buckets or []:
        for p in (b.get("patterns") or []):
            try:
                if re.search(p, s):
                    return b["name"]
            except re.error:
                if p in s:
                    return b["name"]
    return default

def build_feature_meta(feat_cols: list[str], cfg: dict) -> pd.DataFrame:
    type_buckets = (cfg.get("feature_grouping", {}) or {}).get("families", [])
    rows = []
    for f in feat_cols:
        pos_tok = _extract_pos_token(f)
        pos_bucket = _pos_token_to_bucket(pos_tok)
        type_bucket = _match_regex_first(f, type_buckets, default="OtherType")
        rows.append({"feature_id": f, "pos_bucket": pos_bucket, "type_bucket": type_bucket})
    return pd.DataFrame(rows)

# ---------- Controls ----------
def build_controls_Z(df: pd.DataFrame, score_col: str, mode: str="basic") -> pd.DataFrame:
    if mode == "none":
        return pd.DataFrame(index=df.index)
    teacher_cols = [c for c in df.columns if c.startswith("teacher_") and c != score_col]
    size_candidates = [
        "basic_count_word_Cnt","basic_count_sentence_Cnt",
        "sentenceLvl_char_paraLenAvg","sentenceLvl_word_paraLenAvg",
        "essay_len"
    ]
    size_cols = [c for c in size_candidates if c in df.columns]
    Z_cols = list(dict.fromkeys(teacher_cols + size_cols))
    return df[Z_cols].copy()

# ---------- Core stats ----------
def compute_per_feature_stats(df: pd.DataFrame, feat_cols: list[str], score_col: str, is_top: pd.Series, Z: pd.DataFrame):
    y_cont = df[score_col].astype(float).values
    y_bin  = is_top.astype(int).values
    rows = []
    for f in feat_cols:
        x = df[f].astype(float).values
        rows.append({
            "feature_id": f,
            "auc": auc_pos_vs_neg(x, y_bin),
            "d_abs": abs(cohens_d(x[y_bin==1], x[y_bin==0])),
            "spearman_abs": abs(spearman_corr(x, y_cont)),
            "partial_rho_abs": abs(partial_spearman(df[f], df[score_col].astype(float), Z))
        })
    q = pd.DataFrame(rows)
    q["impact"] = (
        zscore(q["auc"].fillna(0.5)) +
        zscore(q["d_abs"].fillna(0.0)) +
        1.5 * zscore(q["partial_rho_abs"].fillna(0.0))
    )
    return q

# ---------- Excel writer ----------
def write_rubric_workbook(out_xlsx: Path, pos_over: pd.DataFrame, type_over: pd.DataFrame,
                          pivot: pd.DataFrame, q_en: pd.DataFrame):
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pos_over.to_excel(writer, sheet_name="POS_overview", index=False)
        type_over.to_excel(writer, sheet_name="TYPE_overview", index=False)
        pivot.to_excel(writer, sheet_name="TYPExPOS")
        # POS sheets
        for pos in sorted(q_en["pos_bucket"].dropna().unique().tolist()):
            dfp = q_en[q_en["pos_bucket"] == pos].sort_values("impact", ascending=False)
            dfp.to_excel(writer, sheet_name=f"POS_{pos[:28]}", index=False)  # Excel sheet name limit
        # TYPE sheets
        for typ in sorted(q_en["type_bucket"].dropna().unique().tolist()):
            dft = q_en[q_en["type_bucket"] == typ].sort_values("impact", ascending=False)
            dft.to_excel(writer, sheet_name=f"TYPE_{typ[:27]}", index=False)
        # Full table (for reference)
        q_en.to_excel(writer, sheet_name="ALL_features", index=False)

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rubrics", nargs="+", required=True,
                    help="List of teacher_* columns, or 'ALL' to auto-detect from config teacher_parser.rubric_names")
    ap.add_argument("--controls", choices=["none","basic"], default="basic")
    ap.add_argument("--balance", choices=["none","subsample"], default="none")
    ap.add_argument("--topn", type=int, default=10, help="Top-N for overview 'impact_topN'")
    args = ap.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load data
    df = load_table(Path(cfg["data_path"]))
    # Teacher parse
    tp = cfg.get("teacher_parser", {}) or {}
    teacher_names = None
    if tp.get("parse", True):
        teacher_names = [s.strip() for s in tp.get("rubric_names", "").split(",")] if tp.get("rubric_names") else None
        df, teacher_cols = attach_teacher_scores(df, tp.get("source_col", "essay_score_avg"), tp.get("delimiter", "#"), teacher_names)

    # Determine rubric columns
    if len(args.rubrics) == 1 and args.rubrics[0].upper() == "ALL":
        rubrics = []
        if teacher_names:
            rubrics = [f"teacher_{name.strip()}" for name in teacher_names]
        else:
            rubrics = [c for c in df.columns if c.startswith("teacher_")]
    else:
        rubrics = args.rubrics

    # Feature pool & meta (shared across rubrics)
    def infer_feature_cols_local(df: pd.DataFrame):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_exact = set(SCORE_COL_EXACT).union(set(df.columns.intersection(META_COLS_HARD)))
        drop_prefixes = SCORE_COL_PREFIXES + TEACHER_COL_PREFIXES
        feat_cols = [c for c in num_cols if c not in drop_exact and not any(c.startswith(p) for p in drop_prefixes)]
        return feat_cols

    feat_cols = infer_feature_cols_local(df)
    meta_df = build_feature_meta(feat_cols, cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for score_col in rubrics:
        if score_col not in df.columns:
            print(f"[WARN] rubric '{score_col}' not in dataframe — skip")
            continue

        # Label: 3.0 positives
        is_top = pd.Series(np.isclose(df[score_col].astype(float), 3.0), index=df.index)

        # Optional balancing
        if args.balance == "subsample":
            pos_idx = np.where(is_top.values == True)[0]
            neg_idx = np.where(is_top.values == False)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                rng = np.random.default_rng(42)
                take_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx)), replace=False)
                keep = np.sort(np.concatenate([pos_idx, take_neg]))
                df_r = df.iloc[keep].reset_index(drop=True)
                is_top_r = is_top.iloc[keep].reset_index(drop=True)
            else:
                df_r, is_top_r = df.copy(), is_top.copy()
        else:
            df_r, is_top_r = df.copy(), is_top.copy()

        # Controls (TYPE features are NOT included in Z)
        Z = build_controls_Z(df_r, score_col, mode=args.controls)

        # Stats across FULL pool
        stats_df = compute_per_feature_stats(df_r, feat_cols, score_col, is_top_r, Z)

        # Merge meta
        q_en = stats_df.merge(meta_df, on="feature_id", how="left")

        # Overviews
        pos_over = (q_en.groupby("pos_bucket", dropna=False)
                      .agg(n=("feature_id","count"),
                           impact_sum=("impact","sum"),
                           impact_mean=("impact","mean"),
                           impact_topN=("impact", lambda s: s.sort_values(ascending=False).head(args.topn).sum()))
                      .reset_index())
        tot = pos_over["impact_sum"].sum() or 1.0
        pos_over["impact_share_pct"] = 100.0 * pos_over["impact_sum"] / tot
        pos_over = pos_over.sort_values("impact_topN", ascending=False)

        type_over = (q_en.groupby("type_bucket", dropna=False)
                       .agg(n=("feature_id","count"),
                            impact_sum=("impact","sum"),
                            impact_mean=("impact","mean"),
                            impact_topN=("impact", lambda s: s.sort_values(ascending=False).head(args.topn).sum()))
                       .reset_index())
        tot2 = type_over["impact_sum"].sum() or 1.0
        type_over["impact_share_pct"] = 100.0 * type_over["impact_sum"] / tot2
        type_over = type_over.sort_values("impact_topN", ascending=False)

        pivot = pd.pivot_table(q_en, values="impact", index="type_bucket", columns="pos_bucket",
                               aggfunc="sum", fill_value=0.0)

        # Write single Excel workbook for this rubric
        safe = score_col.replace("teacher_", "")
        out_xlsx = outdir / f"impact_{safe}.xlsx"
        write_rubric_workbook(out_xlsx, pos_over, type_over, pivot, q_en)
        print(f"[OK] Wrote: {out_xlsx}")

if __name__ == "__main__":
    main()
