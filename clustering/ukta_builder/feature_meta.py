# ukta_builder/feature_meta.py
from __future__ import annotations
import re
import pandas as pd

# ============================================================
# TYPE: 정규식 우선순위(첫 매치만) - 설정의 patterns를 순서대로 검사
# ============================================================
def _match_regex_first(name: str, buckets: list[dict], default: str) -> str:
    """
    config.feature_grouping.families 에 정의된 패턴을 순서대로 검사해
    처음 매칭되는 name을 반환. 매칭 실패 시 default 반환.
    """
    s = name.strip()
    for b in buckets or []:
        for p in (b.get("patterns") or []):
            try:
                if re.search(p, s):
                    return b["name"]
            except re.error:
                # 패턴이 정규식이 아니면 부분문자열 포함으로 폴백
                if p in s:
                    return b["name"]
    return default

# ============================================================
# POS: 세부 토큰 추출 (피처명에서 POS 힌트만 뽑아냄)
# ============================================================
def _extract_pos_token(name: str) -> str | None:
    """
    피처명에서 POS 힌트를 추출한다.
    - 없거나 POS로 보기 힘든 토큰(8-gram/lemma/all/content 등)은 이후 버킷 매핑에서 OTHER 처리됨.
    """
    s = name.strip()

    # 1) NDW_/ttr_ 접두 내 POS
    m = re.search(r"^NDW_([^_]+)_", s) or re.search(r"^ttr_([^_]+)_", s, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2) basic_(count|density|list) 계열
    m = re.search(r"^basic_(count|density|list)_([^_]+)_(Cnt|Den|Lst)$", s, re.IGNORECASE)
    if m:
        return m.group(2)

    # 3) adjacency_* 내 명시적 토큰
    m = re.search(r"overlap_(noun|verb|adjective|adverb|function|content|all|lemma)", s, re.IGNORECASE)
    if m:
        return m.group(1)

    # 4) sentenceLvl_* 계열 (예: sentenceLvl_V_..., sentenceLvl_M_...)
    m = re.search(r"^sentenceLvl_([A-Za-z]+)_", s)
    if m:
        return m.group(1)

    return None

# ============================================================
# POS 토큰 → 표준 버킷 매핑
# ============================================================
def _pos_token_to_bucket(tok: str | None) -> str:
    """
    세부 POS 토큰을 표준 버킷으로 정규화한다.
    표준 버킷: {NOUN, VERB, ADJ, ADV, DET, FUNC, OTHER}
    """
    if tok is None:
        return "OTHER"
    t = tok.strip().upper()

    # 한국어/영문 혼용 태그 처리
    # 명사 계열
    if t in {"NN", "NNG", "NNP", "NNB", "NP", "NR", "N", "NNC", "NNPC", "NNBC", "NMC", "NM", "NC", "NOUN"}:
        return "NOUN"

    # 동사 계열
    if t in {"VV", "VX", "V", "VC", "VERB"}:
        return "VERB"

    # 형용사 계열
    if t in {"VA", "VCN", "VCP", "ADJ", "ADJECTIVE"}:
        return "ADJ"

    # 부사 계열
    if t in {"MA", "MAG", "MAJ", "ADV", "ADVERB"}:
        return "ADV"

    # 관형사/한정사
    if t in {"MM", "M", "MMC", "DET"}:
        return "DET"

    # 기능어/문법요소(조사/어미/보조 등)
    if t in {"J", "E", "X", "F", "IC", "FUNCTION", "FUNC"}:
        return "FUNC"

    # sentenceLvl 단축키류 보정
    if t.startswith("V"):  # V, VA, VV ...
        # V*가 애매하면 동사로 기본 처리
        return "VERB" if t in {"VV", "VX", "V"} else ("ADJ" if t in {"VA", "VCN", "VCP"} else "VERB")
    if t.startswith("N"):
        return "NOUN"
    if t.startswith("M"):  # M은 MM(관형사)일 확률이 높음
        return "DET"

    # POS로 보기 어려운 토큰(8-GRAM, LEMMA, ALL, CONTENT 등) → OTHER
    return "OTHER"

# ============================================================
# 메타 프레임 생성
# ============================================================
def build_feature_meta(feat_cols: list[str], cfg: dict) -> pd.DataFrame:
    """
    - TYPE: config(feature_grouping.families)의 정규식으로 첫 매치만 사용.
    - POS: 피처명에서 POS 힌트를 추출해 표준 버킷으로 정규화.
    반환 컬럼: ['feature_id', 'pos_bucket', 'type_bucket']
    """
    type_buckets = (cfg.get("feature_grouping", {}) or {}).get("families", [])  # TYPE 축 정의

    rows = []
    for f in feat_cols:
        pos_tok = _extract_pos_token(f)
        pos_bucket = _pos_token_to_bucket(pos_tok)
        type_bucket = _match_regex_first(f, type_buckets, default="OtherType")
        rows.append({
            "feature_id": f,
            "pos_bucket": pos_bucket,
            "type_bucket": type_bucket,
        })
    return pd.DataFrame(rows)
