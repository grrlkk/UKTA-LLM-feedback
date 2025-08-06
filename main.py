import argparse
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

# 현재 디렉터리를 모듈 경로에 추가해 prompt.py 임포트가 확실히 되도록
sys.path.append(str(Path(__file__).resolve().parent))

from prompt import create_feedback_prompt_from_ukta_json


def run(json_path: str, model: str = "gpt-4o-mini", temperature: float = 0.4):
    print(f"[DEBUG] json_path: {json_path}, model: {model}, temp: {temperature}")
    """UKTA JSON → GPT 프롬프트 → ChatGPT 호출 → 결과 출력"""
    prompt = create_feedback_prompt_from_ukta_json(json_path)
    print(f"[DEBUG] prompt generated:\n{prompt[:300]}...")  # 너무 길면 일부만

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("❌  OPENAI_API_KEY 가 설정되어 있지 않습니다 (.env 확인)")

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception as e:
        print(f"[ERROR] GPT 호출 실패: {e}")
        return


    print("\n=== GPT 피드백 결과 ===\n")
    print(response.choices[0].message.content.strip())


def cli():
    load_dotenv()  # .env 로부터 환경변수 로드

    parser = argparse.ArgumentParser(description="UKTA JSON → GPT 피드백 생성기")
    parser.add_argument(
        "--file",
        required=True,
        help="UKTA 분석 결과 JSON 경로",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI 모델 이름 (기본: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.4,
        help="GPT temperature (기본: 0.4)",
    )
    args = parser.parse_args()

    run(args.file, model=args.model, temperature=args.temp)


if __name__ == "__main__":
    cli()
