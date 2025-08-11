# prompt.py

## 자질 선정되면 평균, 표준편차 계산 필요

def create_holistic_prompt(ukta_data: dict) -> str:
    """
    UKTA 데이터를 지능적으로 정제/해석하여,
    GPT가 작문 컨설턴트 역할에만 집중하도록 하는 프롬프트를 생성합니다.

    Args:
        ukta_data (dict): 불러온 JSON 데이터 딕셔너리.

    Returns:
        str: GPT API에 전달할 최종 프롬프트 문자열.
    """
    results = ukta_data.get('results', {})
    original_text = results.get('correction', {}).get('origin', '원본 텍스트를 찾을 수 없습니다.')
    scores = results.get('essay_score', {})

    # --- Python이 데이터의 1차 진단까지 완료 ---
    # GPT의 해석 오류를 원천 차단하기 위해, Python 단에서 기준에 따라 질적 진단을 내립니다.
    
    ttr_val = results.get('ttr', {}).get('lemma_TTR', 0)
    ttr_diag = f"적절함 (TTR: {ttr_val:.2f}, 기준치 0.5 이상)" if ttr_val >= 0.5 else f"다소 단조로움 (TTR: {ttr_val:.2f}, 기준치 0.5 미만)"

    sent_len_std_val = results.get('sentenceLvl', {}).get('morph_sentLenStd', 0)
    sent_len_diag = f"다양함 (리듬감 우수, 표준편차: {sent_len_std_val:.1f})" if sent_len_std_val > 5.0 else f"다소 단조로움 (리듬감 비슷, 표준편차: {sent_len_std_val:.1f})"

    sim_val = results.get('similarity', {}).get('avgSentSimilarity', 0)
    sim_diag = f"자연스러움 (문장 간 유사도: {sim_val:.2f}, 기준치 0.3 이상)" if sim_val >= 0.3 else f"연결고리 강화 필요 (문장 간 유사도: {sim_val:.2f}, 기준치 0.3 미만)"

    top_k_features = scores.get('top_k_features', [])

    # --- GPT에게 전달할 '사실 명세서' 생성 ---
    report_text = f"""[UKTA AI 진단 리포트]
### 1. 어휘 구사력
- AI 진단: {ttr_diag}

### 2. 문장 구성력
- AI 진단: {sent_len_diag}

### 3. 논리적 응집성
- AI 진단: {sim_diag}

### 4. 채점 모델이 주목한 핵심 자질
- {', '.join(top_k_features)}
"""

    # --- '근거 인용'과 '구체적 처방'을 강제하는 최종 프롬프트 ---
    prompt = f"""당신은 AI가 분석한 '진단 리포트'의 내용을 학생이 이해하기 쉽게 글로 풀어주는 전문 작문 컨설턴트입니다.
**당신의 임무는 주어진 'AI 진단' 결과를 절대 왜곡하거나 추가 판단 없이, 그대로 인용하여 친절하게 설명하고, 실질적인 개선 방안을 제시하는 것입니다.**

[학생 글 원문]
{original_text}

{report_text}

[피드백 보고서 작성 규칙]
1.  **종합 스타일 진단**: 먼저, 위 리포트의 모든 진단을 종합하여 이 글쓴이의 작문 스타일을 한 문장으로 정의해주세요.
2.  **엄격한 근거 인용 및 심층 설명**: 각 항목(어휘, 문장, 응집성)에 대한 설명을 시작할 때, 반드시 **"AI 진단에 따르면, [진단 결과]라고 합니다. 이는..."** 형식으로, 주어진 진단 결과를 **그대로 복사하여 인용**해야 합니다. 그 후 이것이 작문에서 어떤 의미인지 부드럽게 풀어서 설명해주세요.
3.  **실행 가능한 개선 방안 제시**: 'AI 진단' 결과가 개선이 필요하다고 나온 경우, 반드시 원문에서 **구체적인 예시 문장**을 찾아 "이렇게 바꿔보세요"라고 명확한 수정안을 제시해야 합니다.
4.  **Top-K 자질 기반 종합 의견**: '핵심 자질' 목록을 보고, "AI 채점 모델이 특히 당신의 글에서 ~ 부분을 중요하게 본 것 같습니다. 따라서 ~ 부분을 개선하는 것이 가장 큰 성장을 이끌어낼 것입니다." 와 같이 종합적인 조언을 제공해주세요.
"""
    return prompt