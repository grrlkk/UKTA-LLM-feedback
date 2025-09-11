# ukta_builder/data_loader.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .utils import META_COLS_HARD, SCORE_COL_EXACT, SCORE_COL_PREFIXES

def load_table(path: Path):
    path = Path(path)
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
        print(f"INFO: Teacher source column '{source_col}' not found. Skipping parsing.")
        return df, []
    s = df[source_col].astype(str).fillna("")
    toks = s.str.split(delim)
    maxlen = toks.map(len).max()
    if maxlen == 0:
        return df, []

    n = len(df)
    arr = np.full((n, maxlen), np.nan, dtype=np.float32)
    for i, t in enumerate(toks):
        L = min(len(t), maxlen)
        for j in range(L):
            try:
                val = t[j].strip()
                arr[i, j] = float(val) if val != "" else np.nan
            except (ValueError, TypeError):
                arr[i, j] = np.nan

    names = [f"teacher_{name.strip()}" for name in rubric_names] if rubric_names and len(rubric_names) == maxlen else [f"teacher_rubric_{j}" for j in range(maxlen)]
    for j, name in enumerate(names):
        df[name] = arr[:, j]
    print(f"INFO: Attached teacher columns: {', '.join(names)}")
    return df, names

def infer_feature_cols(df: pd.DataFrame):
    from .utils import TEACHER_COL_PREFIXES
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_exact = SCORE_COL_EXACT | set(df.columns.intersection(META_COLS_HARD))
    drop_prefixes = SCORE_COL_PREFIXES + TEACHER_COL_PREFIXES

    feat_cols = [c for c in num_cols if c not in drop_exact and not any(c.startswith(p) for p in drop_prefixes)]
    return feat_cols

def create_top_mask(df, score_col, top_select, value):
    if top_select == "eq":
        return pd.Series(np.isclose(df[score_col].astype(float), value), index=df.index)
    elif top_select == "quantile":
        if not (0.0 < value < 1.0):
            raise ValueError("q_top must be between 0 and 1 for quantile selection.")
        thr = np.nanpercentile(df[score_col].values, value * 100)
        return df[score_col] >= thr
    else:
        raise ValueError(f"Unsupported top_select mode: {top_select}")