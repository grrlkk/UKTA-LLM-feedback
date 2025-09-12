# clustering/cluster_main.py
import argparse
import warnings
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from joblib import Parallel, delayed  # 병렬화

from ukta_builder.data_loader import (
    load_table, attach_teacher_scores, infer_feature_cols, create_top_mask
)
from ukta_builder.feature_meta import build_feature_meta, _extract_pos_token
from ukta_builder import stats_calculators, feature_selectors, utils

warnings.simplefilter("ignore")


# ---------------------------
# 그리드 셀 내부 경량 스코어 + 대표 선택
# ---------------------------
def _grid_cell_rank(df, feats, score_col, is_top, Z,
                    subsamples=30, C=0.8, partial_rho_min=0.10,
                    rho_thresh=0.90, seed=42,
                    partial_gate=False, partial_min=0.05):
    if len(feats) == 0:
        return pd.DataFrame(columns=["feature_id","auc","d_abs","spearman",
                                     "partial_rho_abs","l1_select_prob","grid_score"])

    if subsamples == 0:
        q = stats_calculators.quick_screen(df, feats, score_col, is_top)  # AUC+|ρ|+|d|
        # 부분상관(게이트용)
        pr = {}
        y = df[score_col].astype(float)
        for f in feats:
            r = feature_selectors.partial_spearman(df[f], y, Z)
            pr[f] = abs(r) if pd.notna(r) else 0.0
        q["partial_rho_abs"] = q["feature_id"].map(pr)

        if partial_gate:
            q = q[q["partial_rho_abs"] >= partial_min].copy()

        # ★ 게이트 후 비었으면 '항상' 빈 DF 반환 (list 금지)
        if q.empty:
            return pd.DataFrame(columns=["feature_id","auc","d_abs","spearman",
                                         "partial_rho_abs","l1_select_prob","grid_score"])

        q["l1_select_prob"] = 0.0
        q.rename(columns={"combo": "grid_score"}, inplace=True)
        return q[["feature_id","auc","d_abs","spearman",
                  "partial_rho_abs","l1_select_prob","grid_score"]]

    # 경량 통계
    stats_df = stats_calculators.discriminative_stats(
        df, feats, score_col, is_top, n_boot=30, topk_rank=20, seed=seed, progress=False
    )
    # 부분상관
    pr = {}
    for f in feats:
        r = feature_selectors.partial_spearman(df[f], df[score_col].astype(float), Z)
        pr[f] = abs(r) if pd.notna(r) else 0.0
    stats_df["partial_rho_abs"] = stats_df["feature_id"].map(pr)

    # L1 안정 선택(경량)
    stab = feature_selectors.l1_logit_stability(
        df[feats].copy(), is_top, C=C, subsamples=subsamples, sample_frac=0.7, seed=seed, standardize=True
    )
    stats_df["l1_select_prob"] = [float(p) for p in stab]

    # 셀 내부 점수
    stats_df["grid_score"] = (
        utils.zscore(stats_df["auc"].fillna(0.5)) +
        utils.zscore(stats_df["d_abs"].fillna(0)) +
        utils.zscore(stats_df["partial_rho_abs"].fillna(0)) +
        0.5 * utils.zscore(stats_df["l1_select_prob"].fillna(0))
    )
    return stats_df


def _select_cell_reps(df, stats_df_cell, rho_thresh=0.90, k=3):
    # ★ 항상 DataFrame을 반환
    if stats_df_cell is None or len(stats_df_cell) == 0:
        return pd.DataFrame(columns=["feature_id","grid_score","family_id"])

    cand = stats_df_cell.sort_values("grid_score", ascending=False)
    fam = feature_selectors.correlation_families(
        df, cand["feature_id"].tolist(), rho_thresh=rho_thresh, progress=False
    )
    if fam is None or fam.empty:
        # 가족이 하나도 안 생겨도 cand에서 상위 k개 그냥 반환
        tmp = cand.head(k).copy()
        tmp["family_id"] = np.arange(len(tmp))
        return tmp[["feature_id","grid_score","family_id"]]

    cand2 = cand.merge(fam, on="feature_id", how="left")
    reps = (cand2.sort_values("grid_score", ascending=False)
                  .groupby("family_id", as_index=False).head(1)
                  .sort_values("grid_score", ascending=False)
                  .head(k))
    # ★ 명시 스키마 보장
    return reps[["feature_id","grid_score","family_id"]]


# ---------------------------
# POS/TYPE/교차 그리드 스크리닝 (병렬)
# ---------------------------
def run_grid(
    df, feat_cols, meta_df, score_col, is_top, Z,
    cfg_grid, cfg_core, outdir: Path, seed=42, n_jobs=8
):
    k = int(cfg_grid.get("top_per_cell", 3))
    min_cell = int(cfg_grid.get("skip_cell_min", 8))
    rho_thresh = float(cfg_core.get("rho_thresh", 0.90))
    mode = cfg_grid.get("mode", "both")  # pos | type | both | pos_type

    # other/OtherType 케이스-무시
    pos_vals = [p for p in meta_df["pos_bucket"].unique() if p and str(p).lower() != "other"]
    type_vals = [t for t in meta_df["type_bucket"].unique() if t and str(t).lower() != "othertype"]

    # 태스크 구성
    tasks = []
    if mode in ("pos", "both"):
        for p in pos_vals:
            feats = meta_df.loc[meta_df["pos_bucket"] == p, "feature_id"].tolist()
            feats = [f for f in feats if f in feat_cols]
            if len(feats) >= min_cell:
                tasks.append(("POS", p, "ANY", f"POS::{p}", feats))

    if mode in ("type", "both"):
        for t in type_vals:
            feats = meta_df.loc[meta_df["type_bucket"] == t, "feature_id"].tolist()
            feats = [f for f in feats if f in feat_cols]
            if len(feats) >= min_cell:
                tasks.append(("TYPE", "ANY", t, f"TYPE::{t}", feats))

    if mode == "pos_type":
        for p in pos_vals:
            for t in type_vals:
                feats = meta_df.loc[
                    (meta_df["pos_bucket"] == p) & (meta_df["type_bucket"] == t), "feature_id"
                ].tolist()
                feats = [f for f in feats if f in feat_cols]
                if len(feats) >= min_cell:
                    tasks.append(("GRID", p, t, f"GRID::{p}x{t}", feats))

    def _run_task(kind, p, t, tag, feats):
        cell = _grid_cell_rank(
            df, feats, score_col, is_top, Z,
            subsamples=0, C=0.8, partial_rho_min=0.10,
            rho_thresh=rho_thresh, seed=seed,
            partial_gate=bool(cfg_grid.get("partial_gate", True)),
            partial_min=float(cfg_grid.get("partial_min", 0.05))
        )
        reps = _select_cell_reps(df, cell, rho_thresh=rho_thresh, k=k)

        out = []
        if reps is None or reps.empty:
            return out  # ★ 빈 DF면 바로 반환

        for _, r in reps.iterrows():
            out.append({
                "feature_id": r["feature_id"],
                "pos_bucket": p,
                "type_bucket": t,
                "source_bucket": tag,
                "grid_score": float(r["grid_score"])
            })
        return out

    # 병렬 실행
    results = Parallel(
        n_jobs=n_jobs, prefer="threads", require="sharedmem", verbose=0
    )(delayed(_run_task)(*task) for task in tasks)
    seeds_rows = [row for lst in results for row in lst]

    seeds_df = pd.DataFrame(seeds_rows).drop_duplicates(subset=["feature_id", "source_bucket"])
    seeds_df.insert(0, "rubric", score_col)
    seeds_df.to_csv(outdir / "anchors_grid.csv", index=False, encoding="utf-8-sig")
    print(f"🧩 anchors_grid.csv saved ({len(seeds_df)} rows)")

    # RRF로 버킷 간 순위 융합 → 최종 seed list
    if seeds_df.empty:
        pd.DataFrame({"feature_id": []}).to_csv(outdir / "seeds_union.csv", index=False, encoding="utf-8-sig")
        return [], seeds_df

    seeds_df["rank"] = seeds_df.groupby("source_bucket")["grid_score"].rank(ascending=False, method="dense")
    seeds_df["rrf"] = 1.0 / (60.0 + seeds_df["rank"])
    fused = (
        seeds_df.groupby("feature_id", as_index=False)["rrf"].sum()
        .sort_values("rrf", ascending=False)
    )
    max_seed = int(cfg_grid.get("final_max_seed", 180))
    seed_list = fused.head(max_seed)["feature_id"].tolist()
    pd.DataFrame({"feature_id": seed_list}).to_csv(outdir / "seeds_union.csv", index=False, encoding="utf-8-sig")
    print(f"🌱 seeds_union.csv saved ({len(seed_list)} seeds)")
    return seed_list, seeds_df


# ---------------------------
# 시드만 결승(비싼 재적합)  — 부분상관 병렬화
# ---------------------------
def final_refit(df, score_col, seeds, is_top, Z, cfg_final, outdir: Path, n_jobs=8):
    if not seeds:
        print("⚠️ final_refit: no seeds; skip")
        return pd.DataFrame(columns=["rubric", "feature_id"])
    seeds = [f for f in seeds if f in df.columns]

    stats_seed = stats_calculators.discriminative_stats(
        df, seeds, score_col, is_top,
        n_boot=60, topk_rank=30, seed=42, progress=False
    )

    # 부분상관 병렬화
    def _ps(f):
        r = feature_selectors.partial_spearman(df[f], df[score_col].astype(float), Z)
        return f, (abs(r) if pd.notna(r) else 0.0)

    pairs = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem", verbose=0)(
        delayed(_ps)(f) for f in seeds
    )
    pr = dict(pairs)

    # L1 안정선택
    stab_arr = feature_selectors.l1_logit_stability(
        df[seeds].copy(), is_top, C=float(cfg_final.get("l1_C", 0.7)),
        subsamples=int(cfg_final.get("l1_subsamples", 120)), sample_frac=0.7, seed=42
    )
    stab = {f: float(p) for f, p in zip(seeds, stab_arr)}

    stats_seed["partial_rho_abs"] = stats_seed["feature_id"].map(pr)
    stats_seed["l1_select_prob"] = stats_seed["feature_id"].map(stab)
    stats_seed["final_score"] = (
        utils.zscore(stats_seed["auc"].fillna(0.5)) +
        utils.zscore(stats_seed["d_abs"].fillna(0)) +
        1.5 * utils.zscore(stats_seed["partial_rho_abs"].fillna(0)) +
        utils.zscore(stats_seed["boot_sign_agree"].fillna(0)) +
        utils.zscore(stats_seed["boot_rank_stability"].fillna(0)) +
        utils.zscore(stats_seed["l1_select_prob"].fillna(0))
    )

    # 결승 단계 최소 문턱
    min_auc = float(cfg_final.get("min_auc", 0.56))
    min_pr = float(cfg_final.get("min_partial_rho", 0.06))
    stats_seed = stats_seed[
        (stats_seed["auc"] >= min_auc) & (stats_seed["partial_rho_abs"] >= min_pr)
    ].copy()
    if len(stats_seed) == 0:
        print(
            f"⚠️ final_refit: all filtered by min gates (AUC>={min_auc}, |partial ρ|>={min_pr}). "
            f"Relax thresholds or check label."
        )
        # 아무것도 없으면 빈 결과 저장 후 반환
        empty = pd.DataFrame(columns=["rubric", "feature_id", "final_score"])
        empty.to_csv(outdir / "anchors_final.csv", index=False, encoding="utf-8-sig")
        return empty

    # (중요) family 묶기 대상 = 문턱 통과한 최종 후보 집합
    fam = feature_selectors.correlation_families(
        df, stats_seed["feature_id"].tolist(),
        rho_thresh=float(cfg_final.get("rho_thresh", 0.85)), progress=False
    )
    rep = stats_seed.merge(fam, on="feature_id", how="left")
    top_rep = (
        rep.sort_values("final_score", ascending=False)
        .groupby("family_id", as_index=False).head(1)
        .sort_values("final_score", ascending=False)
        .head(int(cfg_final.get("keep_top", 12)))
    )

    # 해석 태그(간단히 POS만)
    def _to_pos_bucket(name: str) -> str:
        tok = _extract_pos_token(name)
        return (tok or "OTHER").upper()

    ann = []
    for f in top_rep["feature_id"]:
        ann.append({
            "feature_id": f,
            "pos_bucket": _to_pos_bucket(f),
            "type_bucket": "—"
        })
    ann_df = pd.DataFrame(ann)
    out = top_rep.merge(ann_df, on="feature_id", how="left")
    out.insert(0, "rubric", score_col)
    out.to_csv(outdir / "anchors_final.csv", index=False, encoding="utf-8-sig")
    print(f"🏁 anchors_final.csv saved ({len(out)} rows)")
    return out


def build_two_stage_mask(
    df, total_col: str, total_q: float,
    rubric_col: str, rubric_eq: float,
    min_pos: int = 1000
):
    """
    총점 상위 q(기본 0.80) 컷 내에서 rubric == 특정값(eq, 기본 3.0)만 양성으로 잡는 라벨링.
    양성이 min_pos보다 적으면 q를 0.05씩 낮추며 최대 0.60까지 완화.
    """
    q = float(total_q)

    def make_mask(qcut):
        thr = np.nanpercentile(df[total_col].values, qcut * 100.0)
        m_total = (df[total_col] >= thr)
        m_rub = np.isclose(df[rubric_col].astype(float), rubric_eq)
        return (m_total & m_rub), thr

    mask, thr = make_mask(q)
    while mask.sum() < int(min_pos) and q > 0.60:
        q -= 0.05
        mask, thr = make_mask(q)

    print(f"INFO: Two-stage threshold on {total_col}: q={q:.2f} → thr={thr:.4f}")
    return mask


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Bucket-first selector (POS/TYPE grid)")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--score_col", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=8, help="Parallel workers for grid/final refit")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    show_progress = not args.no_progress
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    print("STEP 1: Loading and preparing data...")
    df = load_table(Path(cfg["data_path"]))
    if cfg.get("downcast", True):
        df = utils.downcast_numeric(df)

    # teacher parse
    tp = cfg.get("teacher_parser", {}) or {}
    if tp.get("parse", True):
        names = [s.strip() for s in tp.get("rubric_names", "").split(",")] if tp.get("rubric_names") else None
        df, _ = attach_teacher_scores(
            df, tp.get("source_col", "essay_score_avg"), tp.get("delimiter", "#"), names
        )

    score_col = args.score_col
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found.")

    # 2) Feature pool & Meta
    feat_cols_all = infer_feature_cols(df)
    print(f"INFO: Found {len(feat_cols_all)} feature columns to analyze.")
    meta_df = build_feature_meta(feat_cols_all, cfg)
    meta_df.to_csv(outdir / "feature_meta.csv", index=False, encoding="utf-8-sig")

    # 3) Top mask (two-stage 라벨링 지원)
    lbl = cfg.get("label", {}) or {}
    if lbl.get("mode") == "two_stage":
        is_top = build_two_stage_mask(
            df,
            total_col=lbl.get("total_col", "essay_scoreT_avg"),
            total_q=lbl.get("total_quantile", 0.80),
            rubric_col=score_col,               # CLI로 들어온 teacher_* (ex. teacher_grammar)
            rubric_eq=lbl.get("rubric_eq", 3.0),
            min_pos=lbl.get("min_pos", 1000),
        )
        print(f"INFO: Two-stage top → positives={int(is_top.sum())} ({is_top.mean()*100:.2f}%)")
    else:
        top_select = cfg.get("top_select", "eq")
        top_value = cfg.get("top_value", cfg.get("q_top", 0.8))
        is_top = create_top_mask(df, score_col, top_select, top_value)
        print(f"INFO: Top group ({top_select} @ {top_value}) contains {is_top.sum()} samples ({is_top.mean()*100:.2f}%).")

    # 4) Controls Z (auto)
    teacher_cols = [c for c in df.columns if c.startswith("teacher_") and c != score_col]
    size_candidates = [
        "basic_count_word_Cnt", "basic_count_sentence_Cnt",
        "sentenceLvl_char_paraLenAvg", "sentenceLvl_word_paraLenAvg",
        "essay_len"
    ]
    basic_controls = [c for c in size_candidates if c in df.columns]
    Z_cols = list(dict.fromkeys(teacher_cols + basic_controls))
    Z = df[Z_cols].copy() if Z_cols else pd.DataFrame(index=df.index)

    # 5) GRID local screening → seeds (병렬 태스크 내부에서 수행)
    print("STEP 2: Grid screening (POS/TYPE buckets)")
    cfg_grid = cfg.get("grid", {
        "enable": True, "local_scoring": True, "top_per_cell": 3,
        "skip_cell_min": 8, "mode": "both", "final_max_seed": 180
    })
    cfg_core = {"rho_thresh": cfg.get("rho_thresh", 0.90)}
    seeds, seeds_df = run_grid(
        df, feat_cols_all, meta_df, score_col, is_top, Z,
        cfg_grid, cfg_core, outdir, seed=cfg.get("seed", 42), n_jobs=args.n_jobs
    )

    # 6) Final refit on seeds (부분상관 병렬)
    if (cfg.get("final", {}) or {}).get("refit", True):
        print("STEP 3: Final refit (expensive on seeds)")
        _ = final_refit(
            df, score_col, seeds, is_top, Z,
            cfg_final=cfg.get("final", {
                "l1_subsamples": 120, "l1_C": 0.7, "rho_thresh": 0.85, "keep_top": 12
            }),
            outdir=outdir, n_jobs=args.n_jobs
        )

    print("\n✅ Analysis Complete!")
    print(f"Artifacts -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
