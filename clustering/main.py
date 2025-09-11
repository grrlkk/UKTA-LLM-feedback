# main.py
import argparse
import warnings
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from ukta_builder import data_loader, stats_calculators, feature_selectors, utils

warnings.simplefilter("ignore")

def main():
    parser = argparse.ArgumentParser(description="UKTA Data-driven Policy Builder")
    parser.add_argument("--config", default="config.yml", help="Path to the config YAML file")
    parser.add_argument("--score_col", required=True, help="Target rubric column to analyze (e.g., teacher_grammar)")
    parser.add_argument("--outdir", required=True, help="Directory to save the output artifacts")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    cli_args = parser.parse_args()

    # 1. 설정 로드
    with open(cli_args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    show_progress = not cli_args.no_progress
    outdir = Path(cli_args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    
    # 2. 데이터 로딩 및 준비
    print("STEP 1: Loading and preparing data...")
    df = data_loader.load_table(config['data_path'])
    if config['downcast']:
        df = utils.downcast_numeric(df)
    
    if config['teacher_parser']['parse']:
        tp_conf = config['teacher_parser']
        names = [s.strip() for s in tp_conf['rubric_names'].split(',')] if tp_conf['rubric_names'] else None
        df, _ = data_loader.attach_teacher_scores(df, tp_conf['source_col'], tp_conf['delimiter'], names)

    if cli_args.score_col not in df.columns:
        raise ValueError(f"Score column '{cli_args.score_col}' not found in the data.")
    
    feat_cols = data_loader.infer_feature_cols(df)
    print(f"INFO: Found {len(feat_cols)} feature columns to analyze.")

    # 3. 상위 그룹 정의
    print("STEP 2: Defining top-performing group...")
    top_val = config.get('top_value', config.get('q_top'))
    is_top = data_loader.create_top_mask(df, cli_args.score_col, config['top_select'], top_val)
    print(f"INFO: Top group ({config['top_select']} @ {top_val}) contains {is_top.sum()} samples ({is_top.mean()*100:.2f}%).")

    # 4. 목표 밴드 계산
    print("STEP 3: Computing target bands for top group...")
    bands_df = stats_calculators.compute_target_bands(df, feat_cols, is_top, progress=show_progress)
    bands_df.to_csv(outdir / "target_bands.csv", index=False, encoding="utf-8-sig")

    # 5. 기본 통계량 및 부트스트랩 안정성 분석
    print("STEP 4: Calculating discriminative stats and bootstrap stability...")
    stats_df = stats_calculators.discriminative_stats(df, feat_cols, cli_args.score_col, is_top, 
                                                      n_boot=config['bootstrap_params']['n_boot'],
                                                      topk_rank=config['bootstrap_params']['topk_rank'],
                                                      seed=config['seed'], progress=show_progress)

    # 6. 데이터 기반 피처 선택 (무거운 연산)
    print("STEP 5: Running advanced feature selectors (this may take a while)...")
    # (a) 부분 상관관계
    teacher_cols = [c for c in df.columns if c.startswith("teacher_") and c != cli_args.score_col]
    Z = df[teacher_cols].copy()
    y_cont = df[cli_args.score_col].astype(float)
    partial_rhos = {f: feature_selectors.partial_spearman(df[f], y_cont, Z) for f in utils._tqdm(feat_cols, desc="Partial Spearman", disable=not show_progress)}
    stats_df["partial_rho_abs"] = stats_df["feature_id"].map(lambda f: abs(partial_rhos.get(f, 0)) if pd.notna(partial_rhos.get(f)) else 0)

    # (b) L1 안정성 선택
    l1_conf = config['l1_params']
    l1_probs_arr = feature_selectors.l1_logit_stability(df[feat_cols], is_top, l1_conf['C'], l1_conf['subsamples'], l1_conf['sample_frac'], config['seed'])
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

    # 8. 상관관계 패밀리 생성 및 앵커 선택
    print("STEP 7: Grouping features into families and choosing anchors...")
    cand_feats = stats_df.loc[stats_df["is_candidate"], "feature_id"].tolist()
    if not cand_feats:
        print("WARNING: No candidates passed filters; falling back to top 60 features by score.")
        cand_feats = stats_df.sort_values("score", ascending=False).head(60)["feature_id"].tolist()
        
    fam_df = feature_selectors.correlation_families(df, cand_feats, config['rho_thresh'], progress=show_progress)
    fam_df.to_csv(outdir / "families.csv", index=False, encoding="utf-8-sig")

    rep_df = stats_df[stats_df["feature_id"].isin(cand_feats)].merge(fam_df, on="feature_id", how="left")
    rep_sorted = rep_df.sort_values("score", ascending=False)
    anchors = rep_sorted.groupby("family_id").head(1)
    
    ac = config['anchor_counts']
    anchors = anchors.sort_values("score", ascending=False).head(ac['max'])
    if len(anchors) < ac['min']:
        needed = ac['min'] - len(anchors)
        extra = rep_sorted[~rep_sorted["feature_id"].isin(anchors["feature_id"])].head(needed)
        anchors = pd.concat([anchors, extra], ignore_index=True)
    
    anchors.insert(0, "rubric", cli_args.score_col)
    anchors.to_csv(outdir / "anchors.csv", index=False, encoding="utf-8-sig")

    print("\n✅ Analysis Complete!")
    print(f"Find your artifacts in: {outdir.resolve()}")

if __name__ == "__main__":
    main()