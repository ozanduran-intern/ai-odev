"""
멀티 에이전트 시스템 - LangGraph 기반 3가지 패턴
1. Supervisor (감독자) 패턴: 감독자가 작업자에게 위임
2. Sequential (순차적) 패턴: 고정 순서로 노드 실행
3. Hierarchical (계층적) 패턴: 상위/하위 감독자 구조

모든 패턴은 ChatOllama (gemma3:4b) 사용
"""

import os
import sys
from typing import TypedDict, Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MODEL_NAME, BASE_URL

# 원본 도구 함수들 직접 임포트 (ReAct 래퍼가 아닌 원본)
from tools.weather import get_weather as _weather_tool
from tools.currency import convert_currency as _currency_tool
from tools.translator import translate as _translate_tool
from tools.place_recommender import recommend_places as _places_tool
from agents.react_agent import search_travel_docs as _rag_tool


# ──────────────────────────────────────────────
# 공통: LLM 헬퍼 함수
# ──────────────────────────────────────────────

def _get_llm() -> ChatOllama:
    """공통 LLM 인스턴스 생성"""
    return ChatOllama(model=MODEL_NAME, base_url=BASE_URL, temperature=0)


def _llm_call(system_prompt: str, user_input: str) -> str:
    """시스템 프롬프트와 사용자 입력으로 LLM 호출"""
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])
    return response.content.strip()


def _extract_city(query: str) -> str:
    """질문에서 도시명 추출 (기본값: seoul)"""
    q = query.lower()
    if "부산" in q or "busan" in q:
        return "busan"
    if "제주" in q or "jeju" in q:
        return "jeju"
    return "seoul"


# ══════════════════════════════════════════════
# 패턴 1: Supervisor (감독자) 패턴
# ══════════════════════════════════════════════

class SupervisorState(TypedDict):
    """감독자 패턴의 상태 정의"""
    messages: Annotated[list, add_messages]  # 대화 기록
    query: str                                # 원본 질문
    next_worker: str                          # 다음 작업자 또는 "FINISH"
    travel_plan: str                          # 여행 계획 결과
    info_result: str                          # 정보 조회 결과
    final_answer: str                         # 최종 답변


def _supervisor_node(state: SupervisorState) -> dict:
    """
    감독자 노드: 상태를 보고 다음 작업자를 결정적으로 선택
    - travel_plan 없음 → travel_planner
    - travel_plan 있고 info_result 없음 → info_assistant (여행에는 날씨/환율도 유용)
    - 둘 다 있음 → FINISH (종합 답변 생성)
    """
    query = state["query"]
    travel_plan = state.get("travel_plan", "")
    info_result = state.get("info_result", "")

    # 1) 양쪽 모두 완료 → 최종 답변 종합
    if travel_plan and info_result:
        final = _llm_call(
            "당신은 여행 어시스턴트입니다. 아래 정보를 종합하여 사용자에게 친절하고 체계적인 최종 답변을 작성하세요.",
            f"질문: {query}\n\n[여행 계획]\n{travel_plan}\n\n[추가 정보]\n{info_result}",
        )
        return {"next_worker": "FINISH", "final_answer": final}

    # 2) 여행 계획이 아직 없음 → 여행 계획자에게 위임
    if not travel_plan:
        return {"next_worker": "travel_planner"}

    # 3) 여행 계획은 있지만 추가 정보 없음 → 정보 도우미에게 위임
    return {"next_worker": "info_assistant"}


def _travel_planner_node(state: SupervisorState) -> dict:
    """
    작업자 1 - 여행 계획자: RAG + 장소 추천 → 여행 계획 작성
    """
    query = state["query"]
    city = _extract_city(query)

    # RAG 문서 검색 + 장소 추천 (맛집, 관광지)
    rag_result = _rag_tool.invoke(query)
    places_food = _places_tool.invoke({"city": city, "category": "food"})
    places_attr = _places_tool.invoke({"city": city, "category": "attractions"})

    # LLM으로 여행 계획 작성
    plan = _llm_call(
        """당신은 한국 여행 전문가입니다.
아래 참고 자료를 기반으로 상세한 여행 계획을 작성하세요.
일정별로 추천 장소, 맛집, 활동을 포함해주세요.
한국어로 작성하세요.""",
        f"질문: {query}\n\n참고 자료:\n{rag_result}\n\n맛집:\n{places_food}\n\n관광지:\n{places_attr}",
    )

    return {
        "travel_plan": plan,
        "messages": [AIMessage(content=f"[여행계획자] 계획 작성 완료")],
    }


def _info_assistant_node(state: SupervisorState) -> dict:
    """
    작업자 2 - 정보 도우미: 날씨 + 환율 정보 수집
    """
    query = state["query"]
    city_kr = "서울"
    for c in ["부산", "제주", "인천", "대구"]:
        if c in query:
            city_kr = c
            break

    # 날씨 조회
    weather = _weather_tool.invoke(city_kr)

    # 환율 정보 (USD, TRY → KRW)
    usd_krw = _currency_tool.invoke({
        "amount": 1.0, "from_currency": "USD", "to_currency": "KRW",
    })

    info = f"[날씨 정보]\n{weather}\n\n[환율 정보]\n{usd_krw}"
    return {
        "info_result": info,
        "messages": [AIMessage(content=f"[정보도우미] 정보 수집 완료")],
    }


def _supervisor_router(state: SupervisorState) -> str:
    """감독자 결정에 따라 다음 노드 선택"""
    next_w = state.get("next_worker", "FINISH")
    if next_w == "FINISH":
        return "finish"
    elif next_w == "info_assistant":
        return "info_assistant"
    return "travel_planner"


def _finish_node(state: SupervisorState) -> dict:
    """최종 답변을 메시지에 추가"""
    final = state.get("final_answer", "작업이 완료되었습니다.")
    return {"messages": [AIMessage(content=final)]}


def create_supervisor_graph():
    """
    감독자 패턴 그래프 생성 및 컴파일
    흐름: START → supervisor → travel_planner → supervisor → info_assistant → supervisor → finish → END
    """
    graph = StateGraph(SupervisorState)

    # 노드 추가
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("travel_planner", _travel_planner_node)
    graph.add_node("info_assistant", _info_assistant_node)
    graph.add_node("finish", _finish_node)

    # 엣지: 시작 → 감독자
    graph.add_edge(START, "supervisor")

    # 감독자 → 조건부 라우팅
    graph.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {"travel_planner": "travel_planner", "info_assistant": "info_assistant", "finish": "finish"},
    )

    # 작업자 → 감독자 복귀
    graph.add_edge("travel_planner", "supervisor")
    graph.add_edge("info_assistant", "supervisor")

    # 종료
    graph.add_edge("finish", END)

    return graph.compile()


# ══════════════════════════════════════════════
# 패턴 2: Sequential (순차적) 패턴
# ══════════════════════════════════════════════

class SequentialState(TypedDict):
    """순차적 패턴의 상태 정의"""
    messages: Annotated[list, add_messages]
    query: str              # 원본 질문
    target_lang: str        # 번역 대상 언어
    research: str           # 연구원의 조사 결과
    plan: str               # 계획자의 계획
    translated: str         # 번역자의 번역 결과
    final_answer: str       # 최종 답변


def _researcher_node(state: SequentialState) -> dict:
    """
    노드 1 - 연구원: RAG 검색 + 장소 추천으로 정보 수집
    """
    query = state["query"]
    city = _extract_city(query)

    # RAG 검색 + 장소 정보 수집
    rag_result = _rag_tool.invoke(query)
    food_info = _places_tool.invoke({"city": city, "category": "food"})
    attr_info = _places_tool.invoke({"city": city, "category": "attractions"})

    research = f"[문서 검색]\n{rag_result}\n\n[맛집]\n{food_info}\n\n[관광지]\n{attr_info}"

    return {
        "research": research,
        "messages": [AIMessage(content=f"[연구원] 조사 완료: {len(research)}자 수집")],
    }


def _planner_node(state: SequentialState) -> dict:
    """
    노드 2 - 계획자: 연구 결과로 여행 가이드 작성
    """
    query = state["query"]
    research = state.get("research", "")

    plan = _llm_call(
        """당신은 여행 가이드 작가입니다.
연구원이 수집한 정보를 바탕으로 체계적인 여행 가이드를 작성하세요.
형식: 1) 개요 2) 추천 일정 3) 맛집 추천 4) 교통 팁
한국어로 작성하세요.""",
        f"요청: {query}\n\n참고 자료:\n{research}",
    )

    return {
        "plan": plan,
        "messages": [AIMessage(content="[계획자] 가이드 작성 완료")],
    }


def _translator_node(state: SequentialState) -> dict:
    """
    노드 3 - 번역자: 가이드를 요청 언어로 번역
    """
    plan = state.get("plan", "")
    target_lang = state.get("target_lang", "en")

    # 질문에서 대상 언어 추출
    lang_map = {"터키어": "tr", "turkish": "tr", "türkçe": "tr",
                "영어": "en", "english": "en", "한국어": "ko", "korean": "ko"}
    query_lower = state.get("query", "").lower()
    for keyword, code in lang_map.items():
        if keyword in query_lower:
            target_lang = code
            break

    # 긴 텍스트는 잘라서 번역 (LLM 컨텍스트 제한)
    text_to_translate = plan[:1500] if len(plan) > 1500 else plan

    translated = _translate_tool.invoke({
        "text": text_to_translate,
        "source_lang": "ko",
        "target_lang": target_lang,
    })

    return {
        "translated": translated,
        "final_answer": translated,
        "messages": [AIMessage(content=f"[번역자] {target_lang}로 번역 완료")],
    }


def create_sequential_graph():
    """
    순차적 패턴 그래프 생성 및 컴파일
    흐름: START → researcher → planner → translator → END
    """
    graph = StateGraph(SequentialState)

    graph.add_node("researcher", _researcher_node)
    graph.add_node("planner", _planner_node)
    graph.add_node("translator", _translator_node)

    # 고정 순서
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "translator")
    graph.add_edge("translator", END)

    return graph.compile()


# ══════════════════════════════════════════════
# 패턴 3: Hierarchical (계층적) 패턴
# ══════════════════════════════════════════════

class HierarchicalState(TypedDict):
    """계층적 패턴의 상태 정의"""
    messages: Annotated[list, add_messages]
    query: str                  # 원본 질문
    travel_done: bool           # 여행팀 작업 완료 여부
    support_done: bool          # 지원팀 작업 완료 여부
    travel_result: str          # 여행팀 결과
    support_result: str         # 지원팀 결과
    final_answer: str           # 최종 답변


def _top_supervisor_node(state: HierarchicalState) -> dict:
    """
    총감독 노드: 결정적 로직으로 팀 배정
    - 양쪽 완료 → 종합 답변 생성
    - 여행팀 미완료 → 여행팀 배정
    - 지원팀 미완료 → 지원팀 배정
    """
    query = state["query"]
    travel_done = state.get("travel_done", False)
    support_done = state.get("support_done", False)
    travel_result = state.get("travel_result", "")
    support_result = state.get("support_result", "")

    # 양쪽 모두 완료 → 종합 답변
    if travel_done and support_done:
        final = _llm_call(
            "당신은 총감독입니다. 양 팀의 보고를 종합하여 사용자에게 완성된 답변을 작성하세요. 한국어로 작성하세요.",
            f"질문: {query}\n\n[여행팀 보고]\n{travel_result}\n\n[지원팀 보고]\n{support_result}",
        )
        return {"final_answer": final}

    return {}


def _travel_team_lead_node(state: HierarchicalState) -> dict:
    """
    하위 감독자 1 - 여행팀장: RAG + 장소 추천 → 여행 계획
    """
    query = state["query"]
    city = _extract_city(query)

    # 정보 수집
    rag_result = _rag_tool.invoke(query)
    food = _places_tool.invoke({"city": city, "category": "food"})
    attractions = _places_tool.invoke({"city": city, "category": "attractions"})
    shopping = _places_tool.invoke({"city": city, "category": "shopping"})

    # 여행 계획 작성
    travel_plan = _llm_call(
        "당신은 여행팀장입니다. 아래 정보를 종합하여 여행 계획을 작성하세요. 추천 일정, 맛집, 관광지, 쇼핑 정보를 포함하세요.",
        f"질문: {query}\n\n문서:\n{rag_result}\n\n맛집:\n{food}\n\n관광지:\n{attractions}\n\n쇼핑:\n{shopping}",
    )

    return {
        "travel_result": travel_plan,
        "travel_done": True,
        "messages": [AIMessage(content="[여행팀장] 여행 계획 작성 완료")],
    }


def _support_team_lead_node(state: HierarchicalState) -> dict:
    """
    하위 감독자 2 - 지원팀장: 날씨 + 환율 정보 수집
    """
    query = state["query"]

    # 도시 추출
    city_kr = "서울"
    for c in ["부산", "제주", "인천", "대구"]:
        if c in query:
            city_kr = c
            break

    # 날씨
    weather = _weather_tool.invoke(city_kr)

    # 환율 (주요 3개 통화)
    usd = _currency_tool.invoke({"amount": 10000.0, "from_currency": "KRW", "to_currency": "USD"})
    try_ = _currency_tool.invoke({"amount": 10000.0, "from_currency": "KRW", "to_currency": "TRY"})
    jpy = _currency_tool.invoke({"amount": 10000.0, "from_currency": "KRW", "to_currency": "JPY"})

    support_info = f"[날씨]\n{weather}\n\n[환율 (10,000 KRW 기준)]\n{usd}\n{try_}\n{jpy}"

    return {
        "support_result": support_info,
        "support_done": True,
        "messages": [AIMessage(content="[지원팀장] 정보 수집 완료")],
    }


def _hierarchical_router(state: HierarchicalState) -> str:
    """총감독 후 라우팅: 결과 유무로 결정적 분기"""
    # 최종 답변이 생성되었으면 종료
    if state.get("final_answer", ""):
        return "compile_final"
    # 여행팀 아직 → 여행팀
    if not state.get("travel_done", False):
        return "travel_team_lead"
    # 지원팀 아직 → 지원팀
    if not state.get("support_done", False):
        return "support_team_lead"
    # 둘 다 끝남 → 종료
    return "compile_final"


def _compile_final_node(state: HierarchicalState) -> dict:
    """최종 답변을 메시지에 기록"""
    final = state.get("final_answer", "")
    if not final:
        # 총감독을 거치지 않고 온 경우 직접 종합
        query = state["query"]
        travel = state.get("travel_result", "정보 없음")
        support = state.get("support_result", "정보 없음")
        final = _llm_call(
            "모든 팀의 보고를 종합하여 최종 답변을 작성하세요.",
            f"질문: {query}\n\n여행팀:\n{travel}\n\n지원팀:\n{support}",
        )
    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }


def _after_travel_team(state: HierarchicalState) -> str:
    """여행팀 완료 후 → 지원팀으로"""
    return "support_team_lead"


def _after_support_team(state: HierarchicalState) -> str:
    """지원팀 완료 후 → 총감독으로 (종합용)"""
    return "top_supervisor"


def create_hierarchical_graph():
    """
    계층적 패턴 그래프 생성 및 컴파일
    흐름: START → top_supervisor → travel_team → support_team → top_supervisor → compile_final → END
    """
    graph = StateGraph(HierarchicalState)

    # 노드 추가
    graph.add_node("top_supervisor", _top_supervisor_node)
    graph.add_node("travel_team_lead", _travel_team_lead_node)
    graph.add_node("support_team_lead", _support_team_lead_node)
    graph.add_node("compile_final", _compile_final_node)

    # 시작 → 총감독
    graph.add_edge(START, "top_supervisor")

    # 총감독 → 조건부 라우팅
    graph.add_conditional_edges(
        "top_supervisor",
        _hierarchical_router,
        {
            "travel_team_lead": "travel_team_lead",
            "support_team_lead": "support_team_lead",
            "compile_final": "compile_final",
        },
    )

    # 여행팀 → 지원팀 (순차)
    graph.add_edge("travel_team_lead", "support_team_lead")

    # 지원팀 → 총감독 (종합)
    graph.add_edge("support_team_lead", "top_supervisor")

    # 종료
    graph.add_edge("compile_final", END)

    return graph.compile()


# ══════════════════════════════════════════════
# 테스트 실행
# ══════════════════════════════════════════════

def run_tests():
    """3개 패턴 모두 테스트"""

    print("=" * 70)
    print("  멀티 에이전트 시스템 - 3가지 패턴 테스트")
    print("=" * 70)

    # ── 패턴 1: Supervisor 테스트 ──
    print(f"\n{'━' * 70}")
    print("  패턴 1: Supervisor (감독자)")
    print("  질문: 서울 3일 여행 계획 세워줘")
    print(f"{'━' * 70}\n")

    try:
        g1 = create_supervisor_graph()
        r1 = g1.invoke({
            "messages": [HumanMessage(content="서울 3일 여행 계획 세워줘")],
            "query": "서울 3일 여행 계획 세워줘",
            "next_worker": "", "travel_plan": "", "info_result": "", "final_answer": "",
        })
        a1 = r1.get("final_answer") or r1["messages"][-1].content
        print(f"✅ [Supervisor] 응답 ({len(a1)}자):\n{a1[:600]}")
    except Exception as e:
        print(f"❌ 오류: {e}")

    # ── 패턴 2: Sequential 테스트 ──
    print(f"\n{'━' * 70}")
    print("  패턴 2: Sequential (순차적)")
    print("  질문: 부산 여행 가이드를 터키어로 만들어줘")
    print(f"{'━' * 70}\n")

    try:
        g2 = create_sequential_graph()
        r2 = g2.invoke({
            "messages": [HumanMessage(content="부산 여행 가이드를 터키어로 만들어줘")],
            "query": "부산 여행 가이드를 터키어로 만들어줘",
            "target_lang": "tr", "research": "", "plan": "", "translated": "", "final_answer": "",
        })
        a2 = r2.get("translated") or r2.get("final_answer") or r2["messages"][-1].content
        print(f"✅ [Sequential] 응답 ({len(a2)}자):\n{a2[:600]}")
    except Exception as e:
        print(f"❌ 오류: {e}")

    # ── 패턴 3: Hierarchical 테스트 ──
    print(f"\n{'━' * 70}")
    print("  패턴 3: Hierarchical (계층적)")
    print("  질문: 제주도 여행 계획 + 환율 정보 + 날씨")
    print(f"{'━' * 70}\n")

    try:
        g3 = create_hierarchical_graph()
        r3 = g3.invoke({
            "messages": [HumanMessage(content="제주도 여행 계획 + 환율 정보 + 날씨")],
            "query": "제주도 여행 계획 + 환율 정보 + 날씨",
            "travel_done": False, "support_done": False,
            "travel_result": "", "support_result": "", "final_answer": "",
        })
        a3 = r3.get("final_answer") or r3["messages"][-1].content
        print(f"✅ [Hierarchical] 응답 ({len(a3)}자):\n{a3[:600]}")
    except Exception as e:
        print(f"❌ 오류: {e}")

    print(f"\n{'=' * 70}")
    print("  모든 패턴 테스트 완료")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_tests()
