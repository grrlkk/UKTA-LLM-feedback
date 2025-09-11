# ukta_builder/utils.py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Meta / Score columns to exclude from feature pool
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
