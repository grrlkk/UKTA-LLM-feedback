import json

def create_feedback_prompt_from_ukta_json(file_path: str) -> str:
    """
    UKTA JSON 분석 결과 파일에서 데이터를 읽어 GPT 피드백 생성용 프롬프트를 만듭니다.

    Args:
        file_path (str): UKTA 분석 결과 JSON 파일의 경로.

    Returns:
        str: GPT API에 전달할 수 있도록 가공된 최종 프롬프트 문자열.
    """
    # 1. JSON 파일 읽기 및 기본 데이터 구조 확인
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"오류: {file_path} 에서 파일을 찾을 수 없습니다."
    except json.JSONDecodeError:
        return f"오류: {file_path} 파일이 올바른 JSON 형식이 아닙니다."

    results = data.get('results', {})
    if not results:
        return "오류: JSON 파일에 'results' 키가 없습니다."

    # 2. 피드백 생성을 위한 핵심 정보 추출
    original_text = results.get('correction', {}).get('origin', '원본 텍스트를 찾을 수 없습니다.')
    
    # 맞춤법 및 띄어쓰기 교정 정보 추출
    corrections = []
    revised_sentences = results.get('correction', {}).get('revisedSentences', [])
    for sentence in revised_sentences:
        if 'revisedBlocks' in sentence:
            for block in sentence['revisedBlocks']:
                # 시스템이 제안하는 첫 번째 교정안을 사용합니다.
                if block.get('revisions'):
                    original_word = block.get('origin', {}).get('content', '')
                    revised_word = block['revisions'][0].get('revised', '')
                    reason = block['revisions'][0].get('comment', '')
                    corrections.append(f"- 원문 '{original_word}' → 추천 '{revised_word}' ({reason})")
    
    # 주요 지표 및 점수 추출
    scores = results.get('essay_score', {})
    ttr = results.get('ttr', {})
    similarity = results.get('similarity', {})
    
    # 3. 추출된 정보를 바탕으로 데이터 요약 부분 구성
    summary_parts = [
        "1. 영역별 점수 (5점 만점):",
        f"   - 문법 및 표현: {scores.get('grammar', 'N/A')}점",
        f"   - 어휘 사용: {scores.get('vocabulary', 'N/A')}점",
        f"   - 글의 구조 및 일관성: {scores.get('structural_consistency', 'N/A')}점",
        "\n2. 주요 언어 지표:",
        f"   - 어휘 다양성 (TTR): {ttr.get('lemma_TTR', 0):.2f} (해석: 0.5 이상이면 다양한 어휘를 사용한 것으로 볼 수 있음)",
        f"   - 문장 간 유사도 (응집성): {similarity.get('avgSentSimilarity', 0):.2f} (해석: 0.3 이상이면 문장 간 연결이 자연스러움)",
        "\n3. 맞춤법 및 표현 교정 제안:"
    ]
    if corrections:
        summary_parts.extend(corrections)
    else:
        summary_parts.append("   - 특별한 교정 사항이 발견되지 않았습니다. 훌륭합니다!")
        
    summary_text = "\n".join(summary_parts)

    # 4. 최종 GPT 프롬프트 조립
    prompt = f"""당신은 세계 최고의 한국어 작문 교육 전문가입니다. 아래 주어진 학생의 글과 UKTA 시스템의 분석 데이터를 바탕으로, 학생의 눈높이에 맞춰 긍정적이고 구체적인 피드백을 작성해주세요.

### [학생 글 원문]
{original_text}

### [UKTA 시스템 분석 데이터 요약]
{summary_text}

### [피드백 작성 지시사항]
1.  **총평**: 글의 전반적인 인상과 특징을 간략하게 요약해주세요.
2.  **칭찬할 점**: 분석 데이터의 강점을 근거로 들어 구체적으로 칭찬하며 학생에게 자신감을 심어주세요.
3.  **개선 제안**: 점수가 낮거나 교정 제안이 있는 부분을 중심으로, 단순히 지적하기보다 원문 내용을 예시로 들어 친절하게 설명하고 대안을 제시해주세요.
4.  **어조**: 시종일관 격려하고 지지하는 따뜻한 어조를 유지해주세요.
"""
    return prompt

# --- 코드 실행 예시 ---
if __name__ == '__main__':
    # 분석할 JSON 파일의 경로를 지정합니다.
    # 사용자의 환경에 맞게 파일명을 수정해주세요.
    json_file_path = './data_json/2025-07-22-12_45_58-C100_cohesion.json'
    
    # 함수를 호출하여 최종 프롬프트를 생성합니다.
    gpt_prompt = create_feedback_prompt_from_ukta_json(json_file_path)
    
    # 생성된 프롬프트를 화면에 출력합니다.
    # 이 출력된 문자열을 복사하여 GPT API에 요청(request)으로 보내면 됩니다.
    print("--- 생성된 최종 GPT 프롬프트 ---")
    print(gpt_prompt)