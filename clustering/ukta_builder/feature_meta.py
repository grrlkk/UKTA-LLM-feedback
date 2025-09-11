# /home/chanwoo/GPT_Feedback/clustering/ukta_builder/feature_meta.py
from __future__ import annotations
import re
import pandas as pd

def _match(name: str, buckets: list[dict], default: str) -> str:
    low = name.lower()
    for b in buckets or []:
        for p in (b.get("patterns") or []):
            if p.lower() in low:
                return b["name"]
    return default

def _extract_pos_token(name: str) -> str | None:
    s = name.strip().lower()

    # 1) NDW / TTR 류 접두부에서 POS 추출
    m = re.search(r"^ndw_([^_]+)_", s) or re.search(r"^ttr_([^_]+)_", s)
    if m: return m.group(1).upper()  # 예: NNG, VV ...

    # 2) basic_* 계열 (count/density/list)
    m = re.search(r"^basic_(count|density|list)_([^_]+)_(cnt|den|lst)$", s)
    if m: return m.group(2).upper()

    # 3) adjacency_*에서 noun/verb/adjective/adverb 파싱
    m = re.search(r"overlap_(noun|verb|adjective|adverb)_", s)
    if m: return m.group(1)  # 그대로 유지(임의 변환 X)

    # 4) sentenceLvl_* 계열 (예: sentenceLvl_V_sentLenAvg, sentenceLvl_M_sentLenStd)
    m = re.search(r"^sentencelvl_([a-z]+)_", s)
    if m: return m.group(1).upper()

    # 5) n-gram / lemma / all/content/function
    for k in ["2-gram","3-gram","4-gram","5-gram","6-gram","7-gram","8-gram",
              "_lemma_", "_all_", "_content_", "_function_"]:
        if k in s:
            return k.strip("_")
    return None

def build_feature_meta(feat_cols: list[str], cfg: dict) -> pd.DataFrame:
    """POS는 자동 추출, TYPE은 config(feature_grouping.families) 패턴으로 부여."""
    type_buckets = (cfg.get("feature_grouping", {}) or {}).get("families", [])  # TYPE 축

    rows = []
    for f in feat_cols:
        rows.append({
            "feature_id": f,
            "pos_bucket": _extract_pos_token(f) or "other",
            "type_bucket": _match(f, type_buckets, default="OtherType"),
        })
    return pd.DataFrame(rows)
