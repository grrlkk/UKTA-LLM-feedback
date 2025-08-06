import json

def create_feedback_prompt_from_ukta_json(file_path: str) -> str:
    """
    UKTA JSON 분석 결과 파일에서 데이터를 읽어 GPT 피드백 생성용 프롬프트를 만듭니다.
    """
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

    original_text = results.get('correction', {}).get('origin', '원본 텍스트를 찾을 수 없습니다.')

    # 맞춤법 및 띄어쓰기 교정 정보 추출
    corrections = []
    revised_sentences = results.get('correction', {}).get('revisedSentences', [])
    for sentence in revised_sentences:
        if 'revisedBlocks' in sentence:
            for block in sentence['revisedBlocks']:
                if block.get('revisions'):
                    original_word = block.get('origin', {}).get('content', '')
                    revised_word = block['revisions'][0].get('revised', '')
                    reason = block['revisions'][0].get('comment', '')
                    corrections.append(f"- 원문 '{original_word}' → 추천 '{revised_word}' ({reason})")

    scores = results.get('essay_score', {})
    ttr = results.get('ttr', {})
    similarity = results.get('similarity', {})

    # 요약 텍스트 구성
    summary_parts = [
        "1. 영역별 점수 (5점 만점):",
        f"   - 문법 및 표현: {scores.get('grammar', 'N/A')}점",
        f"   - 어휘 사용: {scores.get('vocabulary', 'N/A')}점",
        f"   - 글의 구조 및 일관성: {scores.get('structural_consistency', 'N/A')}점",
        "\n2. 주요 언어 지표:",
        f"   - 어휘 다양성 (TTR): {ttr.get('lemma_TTR', 0):.2f} (기준: 0.5 이상이면 다양한 어휘 사용으로 판단)",
        f"   - 문장 간 유사도 (응집성): {similarity.get('avgSentSimilarity', 0):.2f} (기준: 0.3 이상이면 연결이 자연스러움)",
        "\n3. 맞춤법 및 표현 교정 제안:"
    ]
    if corrections:
        summary_parts.extend(corrections)
    else:
        summary_parts.append("   - 특별한 교정 사항이 발견되지 않았습니다. 훌륭합니다!")

    summary_text = "\n".join(summary_parts)

    # ✅ 프롬프트에 점수/자질 활용 강조
    prompt = f"""당신은 세계 최고의 한국어 작문 교육 전문가입니다. 아래 주어진 학생의 글과 UKTA 시스템의 분석 데이터를 바탕으로, 학생의 눈높이에 맞춰 긍정적이고 구체적인 피드백을 작성해주세요.

### [학생 글 원문]
{original_text}

### [UKTA 시스템 분석 데이터 요약]
{summary_text}

### [피드백 작성 지시사항]
1. **총평**: 글의 전반적인 인상과 특징을 간략하게 요약해주세요.
2. **칭찬할 점**:
   - 점수가 높은 영역(문법, 어휘, 구조) 또는 TTR, 문장 유사도 등 언어 지표가 긍정적인 부분을 구체적으로 언급해주세요.
   - 학생이 어떤 점을 잘했는지 근거를 바탕으로 칭찬해주세요.
3. **개선 제안**:
   - 점수가 낮은 영역이나 지표(TTR이 낮거나 문장 간 연결이 부자연스러운 경우)를 중심으로 개선점을 제안해주세요.
   - 맞춤법 교정 외에도 어휘의 다양성, 문장 구성, 표현력 등의 측면을 포함해주세요.
   - 가능한 경우 원문 내용을 인용하여 설명하고, 구체적인 대안도 함께 제시해주세요.
4. **어조**: 시종일관 격려하고 지지하는 따뜻한 어조를 유지해주세요.
"""
    return prompt
