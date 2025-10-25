import os
import time
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any

st.set_page_config(page_title="💬 Chatbot Ver1.0", page_icon="💬")

# ---- Sidebar: 설정 ----
with st.sidebar:
    st.header("⚙️ 설정")
    # 우선순위: st.secrets → 환경변수 → 입력창
    default_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="secrets.toml 또는 환경변수에 저장해두면 편해요."
    )
    model = st.selectbox(
        "Model",
        # 필요시 최신 모델명으로 교체/추가하세요.
        options=[
            "gpt-4o-mini",      # 가벼운 기본
            "gpt-4o",           # 더 좋은 품질
            "gpt-3.5-turbo",    # 레거시 예시
        ],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max output tokens", 128, 4096, 1024, 64)
    history_window = st.number_input("History turns to keep", min_value=2, max_value=50, value=12, step=1)
    system_prompt = st.text_area(
        "System prompt (역할/톤 고정)",
        value="You are a helpful, concise assistant. Reply in the user's language.",
        height=120
    )

# ---- 헤더 ----
st.title("💬 Chatbot")
st.write(
    "간단한 Streamlit + OpenAI 챗봇 예제입니다. "
    "API 키는 사이드바에서 입력하거나 `secrets.toml`에 저장할 수 있어요."
)

# ---- 초기 세션 상태 ----
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# ---- 유틸: 히스토리 자르기(토큰 절약 대용) ----
def window_messages(messages: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    """
    시스템 프롬프트 1개 + 최근 k턴(사용자/어시스턴트 페어)을 유지하는 간단한 윈도우링.
    """
    # 시스템 프롬프트는 호출마다 새로 생성하므로 여기서는 최근 대화만 잘라서 반환
    # user/assistant 번갈아 쌓였다는 가정하에 최근 2k개만 유지
    body = messages[-2*k:] if k > 0 else messages
    return body

# ---- 유틸: 스트리밍 생성기 ----
def stream_chat(client: OpenAI, model: str, sys_prompt: str, hist: List[Dict[str, str]]):
    """
    OpenAI Chat Completions 스트리밍을 감싸서 텍스트만 yield.
    """
    # 요청 메시지 구성: 시스템 → 히스토리
    payload_messages = [{"role": "system", "content": sys_prompt}] + hist

    # 스트리밍 호출
    stream = client.chat.completions.create(
        model=model,
        messages=payload_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    # 청크에서 content만 안전 추출
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            # 일부 청크에 content가 없을 수 있음(역할 전환 등) → 무시
            continue

# ---- 기존 메시지 출력 ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- UI: 대화 초기화/내보내기 ----
cols = st.columns(2)
with cols[0]:
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
with cols[1]:
    if st.button("💾 대화 JSON 내보내기"):
        import json
        st.download_button(
            label="다운로드",
            data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
            file_name=f"chat_{int(time.time())}.json",
            mime="application/json"
        )

# ---- 입력창 ----
prompt = st.chat_input("메시지를 입력하세요…")

# ---- 키 없을 때 안내 ----
if not openai_api_key:
    st.info("🔑 사이드바에서 OpenAI API 키를 입력해주세요.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.warning("빈 메시지는 보낼 수 없어요.")
        else:
            # 사용자 메시지 반영 및 표시
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 어시스턴트 응답 스트리밍
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_text = ""

                # 최근 히스토리 윈도우링
                hist = window_messages(st.session_state.messages, history_window)

                try:
                    with st.spinner("생각 중…"):
                        for token in stream_chat(client, model, system_prompt, hist):
                            full_text += token
                            placeholder.markdown(full_text)
                except Exception as e:
                    st.error(f"응답 중 오류가 발생했어요: {e}")
                    full_text = ""

            # 세션에 저장
            if full_text:
                st.session_state.messages.append({"role": "assistant", "content": full_text})
