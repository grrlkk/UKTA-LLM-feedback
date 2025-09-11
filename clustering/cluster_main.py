# main.py
import argparse
import warnings
from pathlib import Path
import yaml
import pandas as pd
import numpy as np  

from ukta_builder.data_loader import (load_table, attach_teacher_scores, infer_feature_cols, create_top_mask)
from ukta_builder.feature_meta import build_feature_meta  
from ukta_builder import stats_calculators, feature_selectors, utils

warnings.simplefilter("ignore")

def run_selection_for_bucket(stats_df, df, bucket_col: str, suffix: str, show_progress: bool, config: dict, outdir: Path, score_col: str):
    """
    버킷별로 상관-패밀리를 만들고 family 대표 1개씩 뽑은 뒤 점수 상위 N을 앵커로 선발.
    families_{suffix}.csv, anchors_{suffix}.csv, bucket_summary_{suffix}.csv 생성.
    """
    from ukta_builder import feature_selectors

    # 1) 후보 피처 목록
    ac = config['anchor_counts']
    cand_feats = stats_df.loc[stats_df["is_candidate"], "feature_id"].tolist()
    if not cand_feats:
        print(f"WARNING[{suffix}]: No candidates passed filters; fallback to top 60 by score.")
        cand_feats = stats_df.sort_values("score", ascending=False).head(60)["feature_id"].tolist()

    # 2) 버킷별 상관-패밀리 구성
    cand_df = stats_df[stats_df["feature_id"].isin(cand_feats)][["feature_id", bucket_col]].copy()
    fam_parts = []
    for bname, sub in cand_df.groupby(bucket_col, dropna=False):
        sub_feats = sub["feature_id"].tolist()
        if len(sub_feats) == 0:
            continue
        fam_sub = feature_selectors.correlation_families(df, sub_feats, config['rho_thresh'], progress=show_progress)
        fam_sub[bucket_col] = bname
        # (bucket, family_id)로 유일 ID 부여
        fam_sub["family_id"] = fam_sub[bucket_col].astype(str) + "::" + fam_sub["family_id"].astype(str)
        fam_parts.append(fam_sub)
    fam_df = pd.concat(fam_parts, ignore_index=True) if fam_parts else pd.DataFrame(columns=['feature_id','family_id',bucket_col])
    fam_df.to_csv(outdir / f"families_{suffix}.csv", index=False, encoding="utf-8-sig")

    # 3) family 대표 1개씩 → 전체 상위 N
    rep_df = stats_df[stats_df["feature_id"].isin(cand_feats)].merge(fam_df, on="feature_id", how="left")
    rep_sorted = rep_df.sort_values("score", ascending=False)
    anchors = rep_sorted.groupby("family_id", dropna=False).head(1)
    anchors = anchors.sort_values("score", ascending=False).head(ac['max']).copy()

    # min 보장
    if len(anchors) < ac['min']:
        extra = rep_sorted[~rep_sorted["feature_id"].isin(anchors["feature_id"])].head(ac['min'] - len(anchors))
        anchors = pd.concat([anchors, extra], ignore_index=True).sort_values("score", ascending=False).head(ac['max'])

    anchors.insert(0, "rubric", score_col)
    anchors.to_csv(outdir / f"anchors_{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"  → saved anchors_{suffix}.csv ({len(anchors)} rows)")

    # 4) 버킷 요약
    def _agg(g):
        return pd.Series({
            "n_feats": g["feature_id"].nunique(),
            "n_candidates": int(g["is_candidate"].sum()),
            "n_anchors": int(g["feature_id"].isin(anchors['feature_id']).sum()),
            "score_top3_mean": g.sort_values("score", ascending=False)["score"].head(3).mean(skipna=True),
            "auc_mean": g["auc"].mean(skipna=True),
            "rho_partial_mean": g["partial_rho_abs"].mean(skipna=True),
            "spec_mean": g["specificity"].mean(skipna=True)
        })
    sum_df = stats_df.groupby(bucket_col).apply(_agg).reset_index()
    sum_df.insert(0, "rubric", score_col)
    sum_df.to_csv(outdir / f"bucket_summary_{suffix}.csv", index=False, encoding="utf-8-sig")

    return anchors[[ "rubric","feature_id","score",bucket_col ]].copy()


def main():
    parser = argparse.ArgumentParser(description="UKTA Data-driven Policy Builder")
    parser.add_argument("--config", default="config.yml", help="Path to the config YAML file")
    parser.add_argument("--score_col", required=True, help="Target rubric column to analyze (e.g., teacher_grammar)")
    parser.add_argument("--outdir", required=True, help="Directory to save the output artifacts")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--run_mode", choices=["pos", "type", "both"], default="both",
                        help="선발 기준: 품사(POS)별, 타입(TYPE: NDW/TTR/Adjacency/...)별, 혹은 둘 다")
    cli_args = parser.parse_args()

    # 1. 설정 로드
    with open(cli_args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    show_progress = not cli_args.no_progress
    outdir = Path(cli_args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 2. 데이터 로딩 및 준비
    print("STEP 1: Loading and preparing data...")
    df = load_table(config['data_path'])
    if config['downcast']:
        df = utils.downcast_numeric(df)

    if config['teacher_parser']['parse']:
        tp_conf = config['teacher_parser']
        names = [s.strip() for s in tp_conf['rubric_names'].split(',')] if tp_conf['rubric_names'] else None
        df, _ = attach_teacher_scores(df, tp_conf['source_col'], tp_conf['delimiter'], names)

    if cli_args.score_col not in df.columns:
        raise ValueError(f"Score column '{cli_args.score_col}' not found in the data.")

    feat_cols = infer_feature_cols(df)
    print(f"INFO: Found {len(feat_cols)} feature columns to analyze.")

    # 3. 상위 그룹 정의
    print("STEP 2: Defining top-performing group...")
    top_val = config.get('top_value', config.get('q_top'))
    is_top = create_top_mask(df, cli_args.score_col, config['top_select'], top_val)
    print(f"INFO: Top group ({config['top_select']} @ {top_val}) contains {is_top.sum()} samples ({is_top.mean()*100:.2f}%).")

    # 2.5. 메타 태깅 + 하이브리드 퍼널(빠른 스크린)
    print("STEP 2.5: Tagging features (POS/TYPE) and hybrid screening...")
    meta_df = build_feature_meta(feat_cols, config)
    meta_df.to_csv(outdir / "feature_meta.csv", index=False, encoding="utf-8-sig")

    hyb = config.get("hybrid", {}) or {}
    if hyb.get("enabled", False):
        q = stats_calculators.quick_screen(df, feat_cols, cli_args.score_col, is_top)
        q = q.merge(meta_df, on="feature_id", how="left")
        axis = hyb.get("screen_axis", ["pos","type"])
        metric = hyb.get("screen_metric", "combo")
        topk = int(hyb.get("screen_topk_per_bucket", 30))
        U = set()
        if "pos" in axis:
            for p in q["pos_bucket"].dropna().unique():
                U.update(q.loc[q["pos_bucket"]==p].nlargest(topk, metric)["feature_id"].tolist())
        if "type" in axis:
            for t in q["type_bucket"].dropna().unique():
                U.update(q.loc[q["type_bucket"]==t].nlargest(topk, metric)["feature_id"].tolist())
        if len(U) < 60:
            extra = q.nlargest(60, metric)["feature_id"].tolist()
            U.update(extra)
        feat_cols = [f for f in feat_cols if f in U]
        print(f"INFO: Hybrid screened to {len(feat_cols)} features.")

    # 4. 목표 밴드 계산
    print("STEP 3: Computing target bands for top group...")
    bands_df = stats_calculators.compute_target_bands(df, feat_cols, is_top, progress=show_progress)
    bands_df.to_csv(outdir / "target_bands.csv", index=False, encoding="utf-8-sig")

    # 5. 기본 통계량 및 부트스트랩 안정성 분석
    print("STEP 4: Calculating discriminative stats and bootstrap stability...")
    stats_df = stats_calculators.discriminative_stats(
        df, feat_cols, cli_args.score_col, is_top,
        n_boot=config['bootstrap_params']['n_boot'],
        topk_rank=config['bootstrap_params']['topk_rank'],
        seed=config['seed'], progress=show_progress
    )

    # 6. 데이터 기반 피처 선택 (부분상관, L1, 특이성)
    print("STEP 5: Running advanced feature selectors (this may take a while)...")
    # (a) 부분 상관관계
    teacher_cols = [c for c in df.columns if c.startswith("teacher_") and c != cli_args.score_col]
    Z = df[teacher_cols].copy()
    y_cont = df[cli_args.score_col].astype(float)
    partial_rhos = {f: feature_selectors.partial_spearman(df[f], y_cont, Z) for f in utils._tqdm(feat_cols, desc="Partial Spearman", disable=not show_progress)}
    stats_df["partial_rho_abs"] = stats_df["feature_id"].map(lambda f: abs(partial_rhos.get(f, 0)) if pd.notna(partial_rhos.get(f)) else 0)

    # (b) L1 안정성 선택
    l1_conf = config['l1_params']
    l1_probs_arr = feature_selectors.l1_logit_stability(
        df[feat_cols], is_top, l1_conf['C'], l1_conf['subsamples'], l1_conf['sample_frac'], config['seed']
    )
    stats_df["l1_select_prob"] = l1_probs_arr

    # (c) 특이성
    specificity_scores = {}
    for f in utils._tqdm(feat_cols, desc="Specificity", disable=not show_progress):
        rho_g = partial_rhos.get(f, np.nan)
        rho_others = [abs(feature_selectors.partial_spearman(df[f], df[tcol].astype(float), Z)) for tcol in teacher_cols]
        max_other = max([r for r in rho_others if pd.notna(r)], default=0.0)
        specificity_scores[f] = (abs(rho_g) - max_other) if pd.notna(rho_g) else np.nan
    stats_df["specificity"] = stats_df["feature_id"].map(specificity_scores)

    # 7. 최종 점수 계산 및 후보 필터링
    print("STEP 6: Scoring features and selecting candidates...")
    w = config['weights']
    stats_df["score"] = (
        utils.zscore(stats_df["auc"].fillna(0.5)) +
        utils.zscore(stats_df["d_abs"].fillna(0)) +
        w['rho_weight'] * utils.zscore(stats_df["partial_rho_abs"].fillna(0)) +
        utils.zscore(stats_df["boot_sign_agree"].fillna(0)) +
        utils.zscore(stats_df["boot_rank_stability"].fillna(0)) +
        w['specificity_weight'] * utils.zscore(stats_df["specificity"].fillna(0))
    )

    f = config['filters']
    cond = (stats_df["partial_rho_abs"].fillna(0) >= f['partial_rho_min']) & \
           (stats_df["l1_select_prob"].fillna(0) >= l1_conf['select_pmin']) & \
           (stats_df["specificity"].fillna(0) >= f['specificity_min'])
    stats_df["is_candidate"] = cond
    stats_df.to_csv(outdir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 8. POS / TYPE 두 런으로 앵커 선발 + 비교
    stats_df = stats_df.merge(meta_df, on="feature_id", how="left")

    anchors_pos = anchors_type = None
    if cli_args.run_mode in ("pos", "both"):
        print("STEP 7-POS: 품사(POS)-루브릭 기준 선발...")
        anchors_pos = run_selection_for_bucket(
            stats_df, df, bucket_col="pos_bucket", suffix="pos",
            show_progress=show_progress, config=config, outdir=outdir, score_col=cli_args.score_col
        )

    if cli_args.run_mode in ("type", "both"):
        print("STEP 7-TYPE: 자질 TYPE(NDW/TTR/...) - 루브릭 기준 선발...")
        anchors_type = run_selection_for_bucket(
            stats_df, df, bucket_col="type_bucket", suffix="type",
            show_progress=show_progress, config=config, outdir=outdir, score_col=cli_args.score_col
        )

    if cli_args.run_mode == "both" and anchors_pos is not None and anchors_type is not None:
        print("STEP 8: POS vs TYPE 결과 비교 파일 생성...")
        ap = set(anchors_pos["feature_id"].tolist())
        at = set(anchors_type["feature_id"].tolist())
        inter = ap & at
        union = ap | at
        cmp_rows = [{
            "rubric": cli_args.score_col,
            "pos_count": len(ap),
            "type_count": len(at),
            "overlap_count": len(inter),
            "jaccard": (len(inter) / len(union)) if len(union) > 0 else 0.0
        }]
        pd.DataFrame(cmp_rows).to_csv(outdir / "compare_pos_vs_type.csv", index=False, encoding="utf-8-sig")

        # 상세 매핑
        adetail = anchors_pos.merge(anchors_type, on="feature_id", how="outer", suffixes=("_pos","_type"))
        adetail.insert(0, "rubric", cli_args.score_col)
        adetail.to_csv(outdir / "compare_details.csv", index=False, encoding="utf-8-sig")

        # 교차 매트릭스(POS x TYPE)
        union_df = pd.concat([
            anchors_pos.assign(source="pos"),
            anchors_type.assign(source="type")
        ], ignore_index=True).merge(meta_df, on="feature_id", how="left")
        cross = union_df.pivot_table(index="pos_bucket", columns="type_bucket", values="feature_id",
                                     aggfunc="nunique", fill_value=0)
        cross.to_csv(outdir / "compare_matrix_pos_x_type.csv", encoding="utf-8-sig")

    print("\n✅ Analysis Complete!")
    print(f"Find your artifacts in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
