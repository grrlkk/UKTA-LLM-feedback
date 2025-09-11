import pandas as pd
import numpy as np

# --- 설정 ---
# 사용자님의 엑셀 파일 경로
EXCEL_FILE_PATH = '/home/chanwoo/GPT_Feedback/data/feat_train_500.xlsx'
# 교사 점수가 들어있는 컬럼 이름
TEACHER_SCORE_COLUMN = 'essay_score_avg'
# 교사 점수 컬럼의 구분자
DELIMITER = '#'
# 루브릭 이름 목록 (명령어에서 사용한 순서와 동일하게)
RUBRIC_NAMES = [
    "grammar", "vocabulary", "sentence_expression",
    "intra_paragraph_structure", "inter_paragraph_structure",
    "structural_consistency", "length", "topic_clarity", "originality",
    "prompt_comprehension", "narrative"
]
# ---

def parse_teacher_scores(df: pd.DataFrame, source_col: str, delim: str, rubric_names: list) -> pd.DataFrame:
    """교사 점수 문자열을 파싱하여 개별 컬럼으로 추가하는 함수"""
    print(f"'{source_col}' 컬럼에서 점수 파싱을 시작합니다...")
    s = df[source_col].astype(str).fillna("")
    toks = s.str.split(delim)
    maxlen = len(rubric_names)
    n = len(df)
    arr = np.full((n, maxlen), np.nan, dtype=np.float32)

    for i, t in enumerate(toks):
        L = min(len(t), maxlen)
        for j in range(L):
            try:
                arr[i, j] = float(t[j])
            except (ValueError, IndexError):
                arr[i, j] = np.nan

    new_cols = [f"teacher_{name.strip()}" for name in rubric_names]
    teacher_df = pd.DataFrame(arr, columns=new_cols, index=df.index)
    df = df.join(teacher_df)
    
    print(f"파싱 완료. {len(new_cols)}개의 교사 점수 컬럼이 추가되었습니다.")
    return df, new_cols

# --- 메인 실행 로직 ---
try:
    print(f"'{EXCEL_FILE_PATH}' 파일을 로딩합니다...")
    header_df = pd.read_excel(EXCEL_FILE_PATH, nrows=0, engine="openpyxl")
    if TEACHER_SCORE_COLUMN not in header_df.columns:
        raise ValueError(f"'{TEACHER_SCORE_COLUMN}' 컬럼을 파일에서 찾을 수 없습니다.")
    df = pd.read_excel(EXCEL_FILE_PATH, engine="openpyxl", usecols=[TEACHER_SCORE_COLUMN])
    print("파일 로딩 완료.")

    df, teacher_cols = parse_teacher_scores(df, TEACHER_SCORE_COLUMN, DELIMITER, RUBRIC_NAMES)

    print("\n--- 11개 루브릭별 기초 통계량 ---")
    # .describe()를 사용하여 주요 통계량 계산
    stats_df = df[teacher_cols].describe().round(2)
    
    # 보기 쉽게 행과 열을 바꿈 (.T)
    print(stats_df.T)
    
    # 결과를 CSV 파일로 저장
    output_filename = 'rubric_score_stats.csv'
    stats_df.T.to_csv(output_filename)
    print(f"\n통계량 테이블이 '{output_filename}' 파일로 저장되었습니다.")


except FileNotFoundError:
    print(f"오류: '{EXCEL_FILE_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")