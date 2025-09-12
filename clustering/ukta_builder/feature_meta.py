# ukta_builder/feature_meta.py
from __future__ import annotations
import re
import pandas as pd

# ----- TYPE: 정규식 우선순위(첫 매치만) -----
def _match_regex_first(name: str, buckets: list[dict], default: str) -> str:
    s = name.strip()
    for b in buckets or []:
        pats = b.get("patterns") or []
        for p in pats:
            try:
                if re.search(p, s):
                    return b["name"]
            except re.error:
                # 패턴이 정규식이 아닐 경우 부분문자열로 폴백
                if p in s:
                    return b["name"]
    return default

# ----- POS: 세부 토큰 추출 -----
def _extract_pos_token(name: str) -> str | None:
    s = name.strip()

    # 1) NDW_/TTR_ 접두 내 POS
    m = re.search(r"^NDW_([^_]+)_", s) or re.search(r"^ttr_([^_]+)_", s, re.IGNORECASE)
    if m: return m.group(1).upper()

    # 2) basic_* 계열
    m = re.search(r"^basic_(count|density|list)_([^_]+)_(Cnt|Den|Lst)$", s, re.IGNORECASE)
    if m: return m.group(2).upper()

    # 3) adjacency_* 내 명시적 토큰
    m = re.search(r"overlap_(noun|verb|adjective|adverb|function|content|all|lemma)", s, re.IGNORECASE)
    if m: return m.group(1).lower()

    # 4) sentenceLvl_* 계열
    m = re.search(r"^sentenceLvl_([A-Za-z]+)_", s)
    if m: return m.group(1).upper()

    return None

# ----- POS 토큰 → 표준 버킷 매핑 -----
def _pos_token_to_bucket(tok: str | None) -> str:
    if tok is None:
        return "OTHER"
    t = tok.upper()
    # 세부 품사 → 대역 버킷
    if t in {"NNG","NNP","NNB","NP","NR","N","NNC","NNPC","NNBC","NMC","NM","NC"} or t == "NOUN":
        return "NOUN"
    if t in {"VV","VX","V","VC"} or t == "VERB":
        return "VERB"
    if t in {"VA","VCN","VCP"} or t == "ADJECTIVE" or t == "ADJ":
        return "ADJ"
    if t in {"MA","MAG","MAJ"} or t == "ADVERB" or t == "ADV":
        return "ADV"
    if t in {"MM"}:
        return "DET"
    if t in {"J","E","X","F","IC"} or t == "FUNCTION":
        return "FUNC"
    # sentenceLvl 내 단축키 (예: V, M 등)
    if t.startswith("V"):  # V, VA, VV...
        return "VERB" if t in {"VV","VX","V"} else ("ADJ" if t in {"VA","VCN","VCP"} else "VERB")
    if t.startswith("N"):
        return "NOUN"
    if t.startswith("M"):
        return "ADV" if t in {"MA","MAG","MAJ"} else "DET"  # M → MM가 많음
    return "OTHER"

def build_feature_meta(feat_cols: list[str], cfg: dict) -> pd.DataFrame:
    """POS는 자동 추출 후 표준 버킷으로 정규화, TYPE은 config의 정규식으로 1차 매칭."""
    type_buckets = (cfg.get("feature_grouping", {}) or {}).get("families", [])  # TYPE 축

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

def _pos_token_to_bucket(tok: str | None) -> str:
    if not tok:
        return "OTHER"
    t = tok.upper()
    # 한국어 형태소 태그 대략 매핑
    if t in {"NN", "NNG", "NNP", "NNB", "NP", "NR", "N"}:
        return "NOUN"
    if t in {"VA"}:
        return "ADJ"
    if t in {"VV", "V", "VX", "VCP", "VCN"}:
        return "VERB"
    if t in {"MA", "MAG", "MAJ"}:
        return "ADV"
    if t in {"MM", "M", "MMC"}:
        return "DET"
    if t in {"J"}:
        return "PARTICLE"
    if t in {"E"}:
        return "ENDING"
    if t in {"X"}:
        return "AUX/OTHER"
    # adjacency_*에서 뽑힌 영어 토큰도 커버
    if t in {"NOUN", "VERB", "ADJECTIVE", "ADVERB"}:
        return t
    return t  # 그 외는 원형 유지
