"""
ReAct 에이전트 (Reasoning + Acting)
- ChatOllama (gemma3:4b) 기반
- 4개 도구 (날씨, 환율, 번역, 장소추천) + RAG 검색 도구
- 대화 기록을 유지하는 메모리 기능 포함

참고: gemma3:4b는 tool calling API를 지원하지 않으므로
      텍스트 기반 ReAct 프롬프트 방식을 사용합니다.
      다중 파라미터 도구는 JSON 문자열 입력을 파싱하는 래퍼로 감쌉니다.
"""

import json
import os
import sys
import time

import re

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 프로젝트 루트를 sys.path에 추가 (모듈 임포트용)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODEL_NAME,
    BASE_URL,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
)

# 원본 도구 함수들을 내부용으로 임포트
from tools.weather import get_weather as _get_weather_orig
from tools.currency import convert_currency as _convert_currency_orig
from tools.translator import translate as _translate_orig
from tools.place_recommender import recommend_places as _recommend_places_orig


# ──────────────────────────────────────────────
# 1. ReAct용 단일 문자열 입력 래퍼 도구
#    (ReAct 에이전트는 도구 입력을 항상 단일 문자열로 전달하므로
#     다중 파라미터 도구는 JSON 파싱 래퍼가 필요)
# ──────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회합니다. 도시 이름을 입력하세요.
    예시: Seoul, 서울, Busan, Istanbul"""
    try:
        return _get_weather_orig.invoke(city.strip())
    except Exception as e:
        return f"날씨 조회 오류: {str(e)}"


@tool
def convert_currency(input_str: str) -> str:
    """통화를 변환합니다. JSON 형식으로 입력하세요.
    형식: {"amount": 100, "from_currency": "USD", "to_currency": "KRW"}
    지원 통화: KRW, TRY, USD, EUR, JPY"""
    try:
        # JSON 파싱 시도
        data = json.loads(input_str)
        amount = float(data["amount"])
        from_cur = str(data["from_currency"])
        to_cur = str(data["to_currency"])
        return _convert_currency_orig.invoke({
            "amount": amount,
            "from_currency": from_cur,
            "to_currency": to_cur,
        })
    except json.JSONDecodeError:
        return "입력 형식 오류. JSON 형식을 사용하세요: {\"amount\": 100, \"from_currency\": \"USD\", \"to_currency\": \"KRW\"}"
    except KeyError as e:
        return f"필수 필드 누락: {e}. 필요한 필드: amount, from_currency, to_currency"
    except Exception as e:
        return f"환율 변환 오류: {str(e)}"


@tool
def translate(input_str: str) -> str:
    """텍스트를 번역합니다. JSON 형식으로 입력하세요.
    형식: {"text": "안녕하세요", "source_lang": "ko", "target_lang": "en"}
    지원 언어: ko(한국어), en(English), tr(Türkçe)"""
    try:
        data = json.loads(input_str)
        text = str(data["text"])
        src = str(data["source_lang"])
        tgt = str(data["target_lang"])
        return _translate_orig.invoke({
            "text": text,
            "source_lang": src,
            "target_lang": tgt,
        })
    except json.JSONDecodeError:
        return "입력 형식 오류. JSON 형식을 사용하세요: {\"text\": \"hello\", \"source_lang\": \"en\", \"target_lang\": \"ko\"}"
    except KeyError as e:
        return f"필수 필드 누락: {e}. 필요한 필드: text, source_lang, target_lang"
    except Exception as e:
        return f"번역 오류: {str(e)}"


@tool
def recommend_places(input_str: str) -> str:
    """도시의 추천 장소를 제공합니다. JSON 형식으로 입력하세요.
    형식: {"city": "seoul", "category": "food"}
    지원 도시: Seoul, Busan, Jeju / 카테고리: food, attractions, shopping, nightlife"""
    try:
        data = json.loads(input_str)
        city = str(data["city"])
        category = str(data["category"])
        return _recommend_places_orig.invoke({
            "city": city,
            "category": category,
        })
    except json.JSONDecodeError:
        return "입력 형식 오류. JSON 형식을 사용하세요: {\"city\": \"seoul\", \"category\": \"food\"}"
    except KeyError as e:
        return f"필수 필드 누락: {e}. 필요한 필드: city, category"
    except Exception as e:
        return f"장소 추천 오류: {str(e)}"


# ──────────────────────────────────────────────
# 2. RAG 검색 도구: PDF 여행 문서에서 정보 검색
# ──────────────────────────────────────────────

def _build_vectorstore() -> Chroma:
    """
    data/ 폴더의 PDF 문서를 로드하고 벡터 스토어를 생성하거나 기존 것을 로드.
    이미 chroma_db가 있으면 재사용, 없으면 PDF를 파싱하여 새로 생성.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    db_path = os.path.join(PROJECT_ROOT, VECTOR_DB_PATH)

    # 기존 벡터 스토어가 있으면 로드
    if os.path.exists(db_path) and os.listdir(db_path):
        print("[RAG] 기존 벡터 스토어를 로드합니다...")
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
        )

    # PDF 문서 로드
    print("[RAG] PDF 문서를 로드하고 벡터 스토어를 생성합니다...")
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(f"data/ 폴더에 PDF 파일이 없습니다: {data_dir}")

    # 문서를 청크 단위로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[RAG] 총 {len(chunks)}개의 청크로 분할 완료")

    # ChromaDB 벡터 스토어 생성 및 저장
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
    print(f"[RAG] 벡터 스토어 생성 완료 → {db_path}")
    return vectorstore


# 벡터 스토어 (지연 초기화)
_vectorstore = None


def _get_vectorstore() -> Chroma:
    """벡터 스토어를 지연 초기화(lazy init)하여 반환"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


@tool
def search_travel_docs(query: str) -> str:
    """Search KOREA-ONLY travel guide documents. Use ONLY for Korea-specific questions.
    Do NOT use this for Japan, Turkey, France, or any other country.
    For other countries, use recommend_places instead."""
    try:
        vectorstore = _get_vectorstore()
        # 유사도 검색으로 관련 문서 조각 가져오기
        results = vectorstore.similarity_search(query, k=RETRIEVER_K)

        if not results:
            return "관련 여행 정보를 찾을 수 없습니다."

        # 검색 결과 포매팅
        output = f"여행 가이드 검색 결과 ({len(results)}건):\n\n"
        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            content = doc.page_content.strip()
            # 내용이 너무 길면 500자로 잘라서 표시
            if len(content) > 500:
                content = content[:500] + "..."
            output += f"[결과 {i}] (출처: {source}, p.{page})\n"
            output += f"{content}\n\n"

        return output.rstrip()

    except FileNotFoundError as e:
        return f"문서 로드 오류: {str(e)}"
    except Exception as e:
        return f"여행 문서 검색 중 오류가 발생했습니다: {str(e)}"


# ──────────────────────────────────────────────
# 3. ReAct 프롬프트 정의 (한국어 + 영어)
# ──────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """입력 텍스트의 언어를 간단히 감지"""
    import unicodedata
    t = text.strip()
    # 한글 포함 여부
    if any('\uAC00' <= c <= '\uD7A3' or '\u3131' <= c <= '\u318E' for c in t):
        return "Korean (한국어)"
    # 일본어
    if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in t):
        return "Japanese (日本語)"
    # 아랍어
    if any('\u0600' <= c <= '\u06FF' for c in t):
        return "Arabic (العربية)"
    # 태국어
    if any('\u0E00' <= c <= '\u0E7F' for c in t):
        return "Thai (ไทย)"
    # 중국어 (한자만)
    if any('\u4E00' <= c <= '\u9FFF' for c in t):
        return "Chinese (中文)"
    # 러시아어 (키릴 문자)
    if any('\u0400' <= c <= '\u04FF' for c in t):
        return "Russian (Русский)"
    # 터키어 특수 문자
    turkish_chars = set('çÇğĞıİöÖşŞüÜ')
    if any(c in turkish_chars for c in t):
        return "Turkish (Türkçe)"
    # 터키어 키워드 감지 (특수 문자 없는 경우 — 2개 이상 매칭 필요)
    turkish_strong = {'merhaba', 'teşekkür', 'nasıl', 'nerede', 'gezilecek',
                      'yerler', 'hava', 'durumu', 'yemek', 'kaç', 'söyle'}
    turkish_weak = {'bir', 'için', 'çok', 'ile', 'ne', 'var', 'bana', 'ver',
                    'eder', 'misin', 'musun', 'bilgi'}
    words = set(t.lower().split())
    strong_hits = len(words & turkish_strong)
    weak_hits = len(words & turkish_weak)
    # 1 güçlü eşleşme veya 2+ zayıf eşleşme = Türkçe
    if strong_hits >= 1 or weak_hits >= 2:
        return "Turkish (Türkçe)"
    # 기본: 영어
    return "English"


REACT_PROMPT = PromptTemplate.from_template(
    """You are a friendly, multilingual travel assistant for the ENTIRE WORLD.
You help with travel to ANY country — Korea, Turkey, Japan, France, Thailand, and anywhere else.

CRITICAL RULE: You MUST reply in {reply_language}. Every word of your Final Answer must be in {reply_language}.

YOUR PRIMARY JOB: Use tools to answer travel-related questions.
- Weather question → MUST use get_weather tool
- Currency question → MUST use convert_currency tool
- Place/food/attraction recommendation for ANY city → MUST use recommend_places tool
- Translation REQUEST (번역, çevir, translate) → MUST use translate tool
- search_travel_docs → ONLY for Korea-specific detailed info. Do NOT use this for other countries.

ONLY for casual chat (greetings, "nasılsın?", "고마워", thanks) → reply directly with Final Answer, no tools.

RESPONSE QUALITY RULES:
1. Variety is mandatory: Mix popular spots with hidden gems / off-the-beaten-path places. Never give the same 3 recommendations twice.
2. Category mix: For general questions (e.g. "What to do in Japan?"), split your answer into categories — 1 culture, 1 nature, 1 food, 1 shopping etc. Don't stay in one category.
3. No repeats: If you already recommended X in this conversation, suggest something different next time. Say "I suggested X earlier, here's something different..."
4. Add personal touch: Include a brief reason WHY each place is special, not just names. Make the user feel like they're getting insider tips from a friend.

You have access to the following tools:

{tools}

Tool names: {tool_names}

Use the following format EXACTLY. Each response must contain ONLY ONE of these two patterns:

Pattern A - Use a tool:
Thought: <your reasoning>
Action: <tool name from [{tool_names}]>
Action Input: <input for the tool>

Pattern B - Give final answer (ONLY when you have all information needed):
Thought: I now know the final answer
Final Answer: <your complete answer>

STRICT RULES:
1. NEVER write both "Action:" and "Final Answer:" in the same response. Pick ONE pattern only.
2. After writing "Action:", STOP. Do NOT write "Final Answer:" or anything else after it.
3. Only use "Final Answer:" when you have ALL the information you need.
4. If the user asks for multiple things, call ALL relevant tools BEFORE giving a Final Answer.
5. For casual conversation, go DIRECTLY to Final Answer without using any tools.

Tool input formats:
- get_weather: Seoul
- convert_currency: {{"amount": 100, "from_currency": "USD", "to_currency": "KRW"}}
- translate: {{"text": "hello", "source_lang": "en", "target_lang": "ko"}}
- recommend_places: {{"city": "seoul", "category": "food"}}
- search_travel_docs: best things to do in Busan

Example - weather (MUST use tool):
Question: 서울 날씨 어때?
Thought: The user asks about weather. I MUST use the get_weather tool.
Action: get_weather
Action Input: Seoul
Observation: 🌤 서울 현재 날씨: 12°C, 맑음
Thought: I now know the final answer
Final Answer: 서울은 현재 12°C에 맑은 날씨예요! 나들이하기 좋은 날이네요.

Example - multi-tool usage:
Question: 서울 날씨 알려주고, 맛집도 추천해줘
Thought: The user wants weather AND food recommendations. I need to call get_weather first.
Action: get_weather
Action Input: Seoul
Observation: ...weather result...
Thought: Now I need food recommendations for Seoul.
Action: recommend_places
Action Input: {{"city": "Seoul", "category": "food"}}
Observation: ...food recommendations...
Thought: I now know the final answer
Final Answer: ...combined weather + food answer...

Example - casual chat (NO tools):
Question: Nasılsın?
Thought: This is casual chat. No tools needed. Reply in Turkish.
Final Answer: İyiyim, teşekkürler! Sen nasılsın?

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


# ──────────────────────────────────────────────
# 4. 에이전트 생성 함수
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 4a. Custom Output Parser
#     모델이 형식 없이 직접 답변하면 AgentFinish로 처리
# ──────────────────────────────────────────────

from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser


class FriendlyReActParser(ReActSingleInputOutputParser):
    """모델이 ReAct 형식 없이 자연어로 답하면 그 텍스트를 최종 답변으로 반환"""

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            return super().parse(text)
        except Exception:
            # "Final Answer:" 없이 바로 텍스트를 출력한 경우
            cleaned = text.strip()
            # "Thought:" 이후의 텍스트만 추출
            if "Thought:" in cleaned:
                cleaned = cleaned.split("Thought:", 1)[1].strip()
            # 남은 prefix 제거
            for prefix in ["I now know the final answer", "Final Answer:"]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            if not cleaned:
                cleaned = text.strip()
            return AgentFinish(
                return_values={"output": cleaned},
                log=text,
            )


# 전체 도구 목록: 4개 래퍼 도구 + RAG 검색 도구
ALL_TOOLS = [get_weather, convert_currency, translate, recommend_places, search_travel_docs]

# 세션별 대화 기록 저장소 (TTL 기반 자동 정리)
_session_histories: dict[str, InMemoryChatMessageHistory] = {}
_session_timestamps: dict[str, float] = {}

MAX_HISTORY_MESSAGES = 10
SESSION_TTL = 3600  # 1시간 후 만료


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """세션 ID별 대화 기록을 반환 (TTL 만료 세션 자동 정리)"""
    now = time.time()
    # 만료된 세션 정리
    expired = [k for k, v in _session_timestamps.items() if now - v > SESSION_TTL]
    for k in expired:
        _session_histories.pop(k, None)
        _session_timestamps.pop(k, None)

    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    _session_timestamps[session_id] = now
    history = _session_histories[session_id]
    if len(history.messages) > MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-MAX_HISTORY_MESSAGES:]
    return history


class _LanguageAwareAgent:
    """언어를 자동 감지해서 프롬프트에 주입하는 에이전트 래퍼"""

    def __init__(self):
        llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL, temperature=0.7)

        agent = create_react_agent(
            llm=llm,
            tools=ALL_TOOLS,
            prompt=REACT_PROMPT,
            output_parser=FriendlyReActParser(),
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        self._with_memory = RunnableWithMessageHistory(
            self._executor,
            _get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def invoke(self, inputs: dict, config=None, **kwargs):
        # 사용자 입력에서 언어 감지 후 주입
        user_input = inputs.get("input", "")
        lang = _detect_language(user_input)
        inputs["reply_language"] = lang
        return self._with_memory.invoke(inputs, config=config, **kwargs)


def create_agent() -> _LanguageAwareAgent:
    """언어 자동 감지 기능이 포함된 ReAct 에이전트 생성"""
    return _LanguageAwareAgent()


# ──────────────────────────────────────────────
# 5. 테스트 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  한국 여행 AI 어시스턴트 - ReAct Agent 테스트")
    print("=" * 60)

    # 에이전트 생성
    agent = create_agent()

    # 세션 설정 (대화 기록 유지를 위해 동일 session_id 사용)
    config = {"configurable": {"session_id": "test-session"}}

    # 테스트 질문 목록
    test_queries = [
        # 테스트 1: 단일 도구 - 날씨
        "서울 날씨 어때?",
        # 테스트 2: 단일 도구 - 환율
        "100 USD는 한국 돈으로 얼마야?",
        # 테스트 3: 멀티 도구 - 날씨 + 장소 추천
        "서울 날씨 알려주고, 맛집도 추천해줘",
        # 테스트 4: 단일 도구 - 번역 (터키어 → 한국어)
        "Merhaba'yı Korece'de nasıl derim?",
        # 테스트 5: RAG 검색 - 부산 여행 정보
        "What are the best things to do in Busan?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"  테스트 {i}: {query}")
        print(f"{'─' * 60}")

        try:
            result = agent.invoke(
                {"input": query},
                config=config,
            )
            print(f"\n✅ 응답:\n{result['output']}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

        print()

    print("=" * 60)
    print("  모든 테스트 완료")
    print("=" * 60)
