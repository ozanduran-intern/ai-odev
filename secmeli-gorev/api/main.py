"""
Travel Assistant API / 여행 도우미 API
- POST /chat: ReAct 에이전트를 사용한 전체 응답
- POST /chat/stream: SSE 스트리밍 응답
- GET /health: 서버 상태 확인
- POST /translate: 빠른 번역 엔드포인트
"""

import json
import os
import sys
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MODEL_NAME, BASE_URL
from agents.react_agent import create_agent, ALL_TOOLS
from tools.translator import translate as _translate_tool, SUPPORTED_LANGS


# ──────────────────────────────────────────────
# FastAPI 앱 초기화
# ──────────────────────────────────────────────

app = FastAPI(
    title="Beroam — Travel Assistant API",
    description="AI-powered travel assistant API with weather, currency, translation, and place recommendations.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8111", "http://127.0.0.1:8111"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ReAct 에이전트 인스턴스 (대화 메모리 포함, 서버 시작 시 1회 생성)
_agent = None


def _get_agent():
    """에이전트 지연 초기화 (첫 요청 시 생성)"""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


# ──────────────────────────────────────────────
# 요청/응답 모델 정의
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    message: str = Field(..., description="사용자 메시지", examples=["서울 날씨 어때?"])
    session_id: str = Field(default="default", description="세션 ID (대화 기록 유지용)")


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    response: str = Field(..., description="에이전트 응답")
    session_id: str = Field(..., description="세션 ID")


class TranslateRequest(BaseModel):
    """번역 요청 스키마"""
    text: str = Field(..., description="번역할 텍스트", examples=["안녕하세요"])
    source_lang: str = Field(..., description="원본 언어 코드", examples=["ko"])
    target_lang: str = Field(..., description="대상 언어 코드", examples=["en"])


class TranslateResponse(BaseModel):
    """번역 응답 스키마"""
    translated_text: str = Field(..., description="번역된 텍스트")
    source_lang: str = Field(..., description="원본 언어 코드")
    target_lang: str = Field(..., description="대상 언어 코드")


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    """서버 상태 확인 엔드포인트"""
    return {"status": "running", "model": MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    ReAct 에이전트를 사용한 전체 응답 엔드포인트
    - 도구(날씨, 환율, 번역, 장소추천, RAG) 자동 선택
    - session_id로 대화 기록 유지
    """
    try:
        agent = _get_agent()
        config = {"configurable": {"session_id": request.session_id}}

        # 에이전트 실행
        result = agent.invoke({"input": request.message}, config=config)

        return ChatResponse(
            response=result["output"],
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 오류: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE(Server-Sent Events) 스트리밍 응답 엔드포인트
    - 에이전트 추론 과정을 실시간 스트리밍
    - 각 단계(Thought, Action, Observation, Final Answer)를 이벤트로 전송
    """
    async def event_generator():
        try:
            agent = _get_agent()
            config = {"configurable": {"session_id": request.session_id}}

            # 시작 이벤트
            yield {
                "event": "start",
                "data": json.dumps({"session_id": request.session_id}, ensure_ascii=False),
            }

            # 에이전트를 동기 실행 후 결과 스트리밍
            # (gemma3:4b는 비동기 스트리밍을 지원하지 않으므로 동기 실행 후 청크 단위 전송)
            result = await asyncio.to_thread(
                agent.invoke,
                {"input": request.message},
                config,
            )

            output = result.get("output", "")

            # 중간 단계가 있으면 전송
            intermediate = result.get("intermediate_steps", [])
            for step in intermediate:
                action, observation = step
                # 도구 호출 이벤트
                yield {
                    "event": "tool_call",
                    "data": json.dumps({
                        "tool": action.tool,
                        "input": action.tool_input,
                    }, ensure_ascii=False),
                }
                # 도구 결과 이벤트
                yield {
                    "event": "tool_result",
                    "data": json.dumps({
                        "tool": action.tool,
                        "result": str(observation)[:500],
                    }, ensure_ascii=False),
                }

            # 최종 응답을 청크 단위로 스트리밍 (자연스러운 스트리밍 효과)
            chunk_size = 50
            for i in range(0, len(output), chunk_size):
                chunk = output[i:i + chunk_size]
                yield {
                    "event": "token",
                    "data": json.dumps({"content": chunk}, ensure_ascii=False),
                }
                await asyncio.sleep(0.02)  # 자연스러운 스트리밍 효과

            # 완료 이벤트
            yield {
                "event": "end",
                "data": json.dumps({
                    "response": output,
                    "session_id": request.session_id,
                }, ensure_ascii=False),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@app.post("/translate", response_model=TranslateResponse)
def translate_text(request: TranslateRequest):
    """
    빠른 번역 엔드포인트 (에이전트 없이 직접 LLM 호출)
    지원 언어: ko(한국어), en(English), tr(Türkçe)
    """
    try:
        src = request.source_lang.strip().lower()
        tgt = request.target_lang.strip().lower()

        # 지원 언어 확인
        if src not in SUPPORTED_LANGS:
            supported = ", ".join(f"{k}({v})" for k, v in SUPPORTED_LANGS.items())
            raise HTTPException(status_code=400, detail=f"지원하지 않는 원본 언어: '{src}'. 지원 언어: {supported}")
        if tgt not in SUPPORTED_LANGS:
            supported = ", ".join(f"{k}({v})" for k, v in SUPPORTED_LANGS.items())
            raise HTTPException(status_code=400, detail=f"지원하지 않는 대상 언어: '{tgt}'. 지원 언어: {supported}")

        if src == tgt:
            raise HTTPException(status_code=400, detail="원본 언어와 대상 언어가 동일합니다.")

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="번역할 텍스트가 비어 있습니다.")

        # LLM 직접 호출로 번역 (에이전트 우회 → 빠른 응답)
        llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL, temperature=0)

        src_name = SUPPORTED_LANGS[src]
        tgt_name = SUPPORTED_LANGS[tgt]

        response = llm.invoke([
            SystemMessage(content=(
                f"You are a professional translator. "
                f"Translate the following text from {src_name} to {tgt_name}. "
                f"Return ONLY the translated text. No explanations, no notes."
            )),
            HumanMessage(content=request.text),
        ])

        translated = response.content.strip()

        return TranslateResponse(
            translated_text=translated,
            source_lang=src,
            target_lang=tgt,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"번역 오류: {str(e)}")


@app.get("/tools")
def list_tools():
    """사용 가능한 도구 목록 반환"""
    return {
        "tools": [
            {"name": t.name, "description": t.description}
            for t in ALL_TOOLS
        ]
    }


@app.get("/")
def root():
    """Web UI 메인 페이지"""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return FileResponse(os.path.join(static_dir, "index.html"))
