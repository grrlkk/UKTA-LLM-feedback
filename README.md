# GPT_Feedback ^~^  
**UKTA 작문 분석 결과를 기반으로 GPT 피드백을 생성하는 LLM 응용 프로그램**  
**인하대학교 KDD 연구실에서 개발합니다~**

---

## 기능 개요

이 프로젝트는 [UKTA 시스템](https://github.com/inhaKDD/UKTA-web)의 작문 분석 결과(JSON)를 받아  
학생 눈높이에 맞춘 **칭찬 + 개선 제안 + 총평**을 포함한 GPT 피드백을 자동 생성합니다.

UKTA 홈페이지: https://ukta.inha.ac.kr/

---

## 설치 방법

### 1. 리포지토리 클론

```bash
git clone https://github.com/grrlkk/GPT_Feedback.git
cd GPT_Feedback
```

### 2. Python 가상환경 생성

```bash
conda create -n ukta_llm python=3.10 -y
conda activate ukta_llm
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. `.env` 파일 생성

루트 디렉토리에 아래와 같이 `.env` 파일을 만들어주세요:

```
OPENAI_API_KEY=sk-xxxx... (본인의 OpenAI 키)
```

---

## 실행 방법

UKTA에서 출력된 JSON 파일을 기반으로 GPT 피드백을 생성합니다:

```bash
python main.py --file ./data_json/파일명.json
```

옵션:

* `--model`: 사용할 OpenAI 모델 이름 (기본: `gpt-4o-mini`)
* `--temp`: GPT temperature 값 (기본: `0.4`)

예시:

```bash
python main.py --file ./data_json/sample.json --model gpt-4 --temp 0.7
```

---

## 파일 구조

```
GPT_Feedback/
├── main.py                # 실행 파일
├── prompt.py              # GPT 프롬프트 생성기
├── requirements.txt       # 패키지 목록
├── .env                   # OpenAI API 키
├── data_json/             # UKTA 결과 JSON 파일들
```

---

## 참고 정보

* 본 프로젝트는 UKTA 시스템의 결과를 LLM 기반으로 **해석하고 설명하는 것**에 초점을 둡니다.
* GPT 프롬프트는 **맞춤법 교정뿐만 아니라**, 점수와 언어 자질(TTR, 유사도 등)을 활용해 학생 맞춤형 피드백을 생성합니다.
