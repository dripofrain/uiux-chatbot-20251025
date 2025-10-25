# streamlit_app.py
import os
import json
import time
import datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import tiktoken
from openai import OpenAI

# 버전 정보
_VERSION = "1.0.0"
_TITLE = "💬 Chatbot "+_VERSION


#st.set_page_config(page_title="💬 Chatbot (Responses API + Tools + tiktoken)", page_icon="🧠")
st.set_page_config(page_title=_TITLE, page_icon="🧠")

# =========================
# Sidebar: Settings
# =========================
with st.sidebar:
    st.header("⚙️ 설정")
    default_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=default_key)

    # Responses API에서 쓸 모델들 (필요시 최신 모델명으로 교체)
    model = st.selectbox(
        "Model (Responses API)",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_output_tokens = st.slider("Max output tokens", 128, 4096, 1024, 64)
    safety_margin_tokens = st.number_input("Safety margin tokens", min_value=256, max_value=8192, value=1024, step=64)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise assistant. Reply in the user's language.",
        height=120
    )
    enable_tools = st.toggle("Enable tool calling (math_sum, get_current_time)", value=True)

#st.title("💬 Chatbot (Responses API + Tools + tiktoken)")
st.title(_TITLE)
st.caption("tiktoken 기반 토큰 카운팅/자동 트렁케이션 + Responses API 스트리밍 + 함수 호출 데모")

# =========================
# Model context window & encodings (heuristic map)
# =========================
# 실서비스에서는 공식 스펙 시트 확인 권장. 여기선 흔한 값들 사용.
MODEL_CONTEXT_MAP = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-3.5-turbo": 16_385,
}
# tiktoken 인코딩 맵(모델 계열별 대체)
# 최신 모델(o/X 계열)은 대부분 cl100k_base로 잘 동작합니다. 맞는 인코더가 없으면 fallback.
MODEL_ENCODER_MAP = {
    "gpt-4o-mini": "cl100k_base",
    "gpt-4o": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

def get_context_window(m: str) -> int:
    return MODEL_CONTEXT_MAP.get(m, 128_000)

def get_encoder_name(m: str) -> str:
    return MODEL_ENCODER_MAP.get(m, "cl100k_base")

def count_tokens_text(text: str, encoder_name: str) -> int:
    enc = tiktoken.get_encoding(encoder_name)
    return len(enc.encode(text))

def count_tokens_messages(messages: List[Dict[str, str]], encoder_name: str) -> int:
    """
    간단한 근사치: role/content를 합쳐 문자열로 붙여 토큰 카운트.
    Responses API는 input에 messages 리스트를 넣지만, 여기선 근사 계산으로 충분.
    """
    # "[role]: content\n" 형태로 직렬화 후 카운트 (메타토큰 약간의 오차 있음)
    blob = ""
    for m in messages:
        blob += f"{m['role']}:\n{m['content']}\n"
    return count_tokens_text(blob, encoder_name)

def truncate_messages_by_tokens(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_msg: Dict[str, str],
    model: str,
    max_output_tokens: int,
    safety_margin: int
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    컨텍스트 윈도우 내에 들어오도록 히스토리를 뒤에서부터 잘라내기.
    - 시스템 프롬프트 + 히스토리(잘라냄) + 사용자 메시지
    - 남겨둬야 할 출력 토큰 + 세이프티 마진 확보
    """
    context_window = get_context_window(model)
    encoder_name = get_encoder_name(model)

    # 우선 전부 포함한 뒤 토큰 계산
    base_messages = [{"role": "system", "content": system_prompt}] + history + [user_msg]
    total_tokens = count_tokens_messages(base_messages, encoder_name)

    # 허용 입력 토큰 = 컨텍스트 - (출력 토큰 + 세이프티 마진)
    max_input_tokens = max(1024, context_window - (max_output_tokens + safety_margin))

    cut_count = 0
    kept_history = history[:]

    while total_tokens > max_input_tokens and kept_history:
        # 가장 오래된 메시지부터 제거
        kept_history.pop(0)
        cut_count += 1
        base_messages = [{"role": "system", "content": system_prompt}] + kept_history + [user_msg]
        total_tokens = count_tokens_messages(base_messages, encoder_name)

    # 최종 메시지
    final_messages = [{"role": "system", "content": system_prompt}] + kept_history + [user_msg]

    stats = {
        "context_window": context_window,
        "max_input_tokens": max_input_tokens,
        "estimated_input_tokens": total_tokens,
        "cut_count": cut_count
    }
    return final_messages, stats

# =========================
# Simple tool implementations (server-side)
# =========================
def tool_math_sum(numbers: List[float]) -> float:
    try:
        return float(sum(numbers))
    except Exception:
        return float("nan")

def tool_get_current_time(tz: str = "Asia/Seoul") -> str:
    # 간단 구현: 타임존 미적용(데모). 실제론 pytz/zoneinfo 사용 권장.
    now = dt.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Responses API에 등록할 도구 스키마
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "math_sum",
            "description": "Return the sum of a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "A list of numbers to sum."
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "description": "IANA timezone string, e.g., Asia/Seoul"}
                },
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

def execute_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """도구를 실제 실행하고 문자열 결과를 반환."""
    if name == "math_sum":
        nums = arguments.get("numbers", [])
        result = tool_math_sum(nums)
        return json.dumps({"sum": result}, ensure_ascii=False)
    if name == "get_current_time":
        tz = arguments.get("tz", "Asia/Seoul")
        result = tool_get_current_time(tz)
        return json.dumps({"now": result, "tz": tz}, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)

# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# 기존 메시지 출력
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 대화 제어 버튼
c1, c2 = st.columns(2)
with c1:
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
with c2:
    if st.button("💾 대화 JSON 내보내기"):
        st.download_button(
            label="다운로드 (아래 버튼 클릭)",
            data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
            file_name=f"chat_{int(time.time())}.json",
            mime="application/json"
        )

# 입력창
user_text = st.chat_input("메시지를 입력하세요…")

# =========================
# Responses API call (streaming + tools)
# =========================
if not openai_api_key:
    st.info("🔑 사이드바에서 OpenAI API 키를 입력해주세요.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    if user_text:
        user_text = user_text.strip()
        if not user_text:
            st.warning("빈 메시지는 보낼 수 없어요.")
        else:
            # 사용자 메시지 반영/표시
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)

            # --- tiktoken 기반 자동 트렁케이션 ---
            # Responses API는 input에 messages를 그대로 넣을 수 있으므로,
            # 여기선 시스템 프롬프트 + 히스토리 + 현재 사용자 메시지만 포함.
            history_only = [m for m in st.session_state.messages[:-1]]  # 마지막은 현재 user
            current_user = {"role": "user", "content": user_text}
            final_messages, tok_stats = truncate_messages_by_tokens(
                system_prompt=system_prompt,
                history=history_only,
                user_msg=current_user,
                model=model,
                max_output_tokens=max_output_tokens,
                safety_margin=safety_margin_tokens,
            )

            # UI로 토큰 통계 표시(선택)
            with st.expander("토큰/컨텍스트 통계 보기", expanded=False):
                st.json(tok_stats)

            # --- Responses API 스트리밍 호출 ---
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_text = ""

                try:
                    # Responses API는 input에 messages 포맷을 그대로 줄 수 있음
                    # tool 사용 허용 시 tools 전달
                    stream_kwargs = dict(
                        model=model,
                        input=final_messages,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    if enable_tools:
                        stream_kwargs["tools"] = TOOLS_SPEC
                        stream_kwargs["tool_choice"] = "auto"  # 모델이 필요 시 도구 호출

                    # 1) 1차 스트림: 필요 시 tool 호출 이벤트가 나옴
                    with client.responses.stream(**stream_kwargs) as stream:
                        for event in stream:
                            # 텍스트 델타
                            if event.type == "response.output_text.delta":
                                delta = event.delta
                                if delta:
                                    full_text += delta
                                    placeholder.markdown(full_text)
                            # 도구 호출 요청 이벤트
                            elif event.type == "response.tool_call.delta":
                                # Responses API는 스트림 이후에 전체 응답 객체에서 tool_calls를 정리해서 주기 때문에
                                # 여기서는 별도 처리를 하지 않고 종료 후 후처리로 일괄 처리한다.
                                pass
                        # 스트림 종료 후 최종 응답(메타 포함)
                        first_response = stream.get_final_response()

                    # 2) 도구 호출이 필요한 경우: tool_calls 파싱 → 도구 실행 → tool_outputs로 재호출
                    tool_calls = []
                    try:
                        # Responses API의 구조에서 tool 호출 정보 추출
                        if hasattr(first_response, "output") and first_response.output:
                            for item in first_response.output:
                                if item.type == "tool_call":
                                    tool_calls.append(item)
                    except Exception:
                        # 일부 SDK 버전/응답 형식 차이에 대비한 안전장치
                        pass

                    if enable_tools and tool_calls:
                        # 도구 실행 결과 모으기
                        tool_outputs = []
                        for call in tool_calls:
                            name = call.name
                            arguments = call.arguments if isinstance(call.arguments, dict) else {}
                            result = execute_tool_call(name, arguments)
                            tool_outputs.append({
                                "tool_call_id": call.id,
                                "output": result
                            })

                        # tool_outputs를 첨부해 최종 답변 생성(비동기 X, 즉시 재호출)
                        final = client.responses.create(
                            model=model,
                            input=final_messages,  # 같은 입력 유지
                            tool_choice="none",    # 이제는 도구 호출 안 함
                            tool_outputs=tool_outputs,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        )

                        # 최종 텍스트를 덮어쓰기/추가
                        final_text = ""
                        if hasattr(final, "output_text"):
                            final_text = final.output_text
                        elif hasattr(final, "output") and final.output:
                            # 일부 SDK 호환
                            chunks = []
                            for item in final.output:
                                if getattr(item, "type", "") == "output_text":
                                    chunks.append(getattr(item, "content", ""))
                            final_text = "".join(chunks)

                        if final_text:
                            full_text = final_text
                            placeholder.markdown(full_text)

                except Exception as e:
                    st.error(f"응답 중 오류가 발생했어요: {e}")
                    full_text = ""

            # 세션에 저장
            if full_text:
                st.session_state.messages.append({"role": "assistant", "content": full_text})
