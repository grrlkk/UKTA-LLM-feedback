# main.py

import os
import json
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 불러옵니다
load_dotenv()

# OpenAI API 클라이언트 설정
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"OpenAI API 키 설정에 문제가 발생했습니다: {e}")
    client = None

# prompt.py 파일에서 프롬프트 생성 함수를 가져옵니다.
from prompt import create_holistic_prompt

def call_gpt_api(prompt: str, model: str, temp: float) -> str:
    """GPT API를 호출하여 피드백을 생성합니다."""
    if not client or not client.api_key:
        return "오류: OpenAI 클라이언트가 초기화되지 않았습니다. .env 파일에 OPENAI_API_KEY가 올바르게 설정되었는지 확인해주세요."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful writing consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT API 호출 중 오류 발생: {e}"

def save_feedback(feedback_content: str, model_name: str, directory: str = "./results"):
    """
    생성된 피드백을 새로운 규칙에 따라 마크다운 파일로 저장합니다.
    (예: fb_250807_gpt-4o_1.md)
    """
    try:
        os.makedirs(directory, exist_ok=True)
        
        # 날짜(YYMMDD)와 모델명으로 기본 파일명 생성
        date_str = datetime.now().strftime("%y%m%d")
        # 모델명에 포함될 수 있는 '/' 문자를 '_'로 변경하여 파일명 오류 방지
        sanitized_model_name = model_name.replace("/", "_")
        base_filename = f"fb_{date_str}_{sanitized_model_name}"
        
        # 동일한 이름의 파일이 있을 경우, 순번을 증가시켜 중복 방지
        index = 1
        while True:
            file_name = f"{base_filename}_{index}.md"
            file_path = os.path.join(directory, file_name)
            if not os.path.exists(file_path):
                break
            index += 1
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(feedback_content)
            
        print(f"✅ 피드백이 성공적으로 저장되었습니다: {file_path}")
    except Exception as e:
        print(f"❌ 파일 저장 중 오류가 발생했습니다: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="UKTA 분석 결과로 GPT 피드백을 생성하고 저장합니다.")
    parser.add_argument("--file", required=True, help="분석할 UKTA JSON 파일의 경로")
    parser.add_argument("--model", default="gpt-4o", help="사용할 OpenAI 모델")
    parser.add_argument("--temp", type=float, default=0.3, help="생성 시 Temperature 값")
    args = parser.parse_args()

    print(f"[INFO] 파일 경로: {args.file}, 모델: {args.model}")

    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            ukta_data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    final_prompt = create_holistic_prompt(ukta_data)
    print("[INFO] 프롬프트 생성 완료.")
    
    print("[INFO] GPT API 호출 중...")
    gpt_feedback = call_gpt_api(final_prompt, args.model, args.temp)
    print("[INFO] GPT 피드백 생성 완료.")
    
    # save_feedback 함수에 model 이름을 인자로 전달
    save_feedback(gpt_feedback, model_name=args.model)
    
    print("\n=== 최종 피드백 결과 ===")
    print(gpt_feedback)

if __name__ == '__main__':
    main()