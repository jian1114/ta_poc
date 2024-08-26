import streamlit as st
import time  # 시간을 사용해 고유 키 생성
import os
import openai
import asyncio
openai.api_key = st.secrets["OPENAI_API_KEY"]



# Streamlit 앱 제목 설정
st.title('GPT 기반 상담 분류기 - 동시 결과 및 스트리밍')

# GPT 모델 목록
models = ['gpt-4o', 'gpt-4o-mini']

default_prompt = """#목적
상담 내용을 바탕으로, 아래에 주어진 15개의 상담유형 리스트 중에서 가장 유사한 3가지 상담유형을 추천해주세요.

#상담유형 리스트
'서비스문의>제품문의'
'기타안내>대리점안내'
'도비도스몰>구매문의'
'서비스접수>독촉'
'서비스문의>사용설명서요청'
'기타안내>카달로그/성적서/인증서'
'서비스문의>접수내용확인'
'도비도스몰>설치문의'
'서비스문의>부품구매'
'서비스접수>서비스접수'
'내선연결'
'서비스문의>타사품'
'서비스접수>접수취소'
'서비스문의>이전설치'
'서비스문의>기타'

#출력조건
설명 없이, 예시와 같은 형식으로 결과만 출력하세요.

#예시 출력
서비스문의>제품문의, 기타안내>대리점안내, 도비도스몰>구매문의
"""

default_input = """안녕하세요
"""

prompt = st.text_area('GPT Prompt 입력', value=default_prompt, height=200)

# GPT Input 입력
user_input = st.text_area('GPT Input 입력', value=default_input, height=100)


# GPT 파라미터 선택
temperature = st.slider('Temperature', 0.0, 1.0, 0.7)
max_tokens = st.slider('Max Tokens', 10, 1000, 200)
top_p = st.slider('Top P', 0.0, 1.0, 1.0)
frequency_penalty = st.slider('Frequency Penalty', 0.0, 2.0, 0.0)
presence_penalty = st.slider('Presence Penalty', 0.0, 2.0, 0.0)

# Streamlit placeholders for dynamic content update
placeholders = {model: st.empty() for model in models}

async def fetch_gpt_response(model, prompt, user_input):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True 
        )

        result_text = ""
        for chunk in response:
            chunk_text = chunk['choices'][0]['delta'].get('content', '')
            result_text += chunk_text
            # 고유한 키 생성: 모델 이름과 현재 시간 조합
            unique_key = f'{model}_result_{hash(result_text)}_{time.time()}'
            placeholders[model].text_area(f'{model} 결과', value=result_text, height=200, key=unique_key)

    except Exception as e:
        placeholders[model].error(f"{model} 오류 발생: {e}")

async def main():
    tasks = [fetch_gpt_response(model, prompt, user_input) for model in models]
    await asyncio.gather(*tasks)


if st.button('결과 생성'):
    asyncio.run(main())
