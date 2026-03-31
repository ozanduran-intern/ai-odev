"""
전체 테스트 스위트
1. ReAct 에이전트 (5개 시나리오)
2. 멀티 에이전트 3패턴 (Supervisor, Sequential, Hierarchical)
3. 검색기 3종 비교 (Keyword, Vector, Hybrid)
4. FastAPI 엔드포인트 (/health, /chat, /translate, /chat/stream)
"""

import os
import sys
import time
import json
import subprocess
import signal

import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0
SKIP = 0


def result(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    suffix = f" - {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return ok


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ──────────────────────────────────────────────
# 1. ReAct 에이전트 테스트
# ──────────────────────────────────────────────

def test_react_agent():
    section("1. ReAct Agent Tests")
    from agents.react_agent import create_agent

    agent = create_agent()
    config = {"configurable": {"session_id": "test-react"}}

    cases = [
        ("Weather single tool", "서울 날씨 어때?", lambda r: "12" in r or "Clear" in r or len(r) > 10),
        ("Currency single tool", "100 USD는 한국 돈으로 얼마야?", lambda r: "133" in r or "KRW" in r or "won" in r.lower()),
        ("Multi tool (weather+food)", "서울 날씨 알려주고, 맛집도 추천해줘", lambda r: len(r) > 20),
        ("Translate Turkish->Korean", "Merhaba'yı Korece'de nasıl derim?", lambda r: "안녕" in r),
        ("RAG search Busan", "What are the best things to do in Busan?", lambda r: len(r) > 20),
    ]

    for name, query, check in cases:
        try:
            res = agent.invoke({"input": query}, config=config)
            out = res.get("output", "")
            result(name, check(out), f"{len(out)} chars")
        except Exception as e:
            result(name, False, str(e)[:80])


# ──────────────────────────────────────────────
# 2. 멀티 에이전트 테스트
# ──────────────────────────────────────────────

def test_multi_agent():
    section("2. Multi-Agent Pattern Tests")
    from langchain_core.messages import HumanMessage

    # Pattern 1: Supervisor
    try:
        from agents.multi_agent import create_supervisor_graph
        g = create_supervisor_graph()
        r = g.invoke({
            "messages": [HumanMessage(content="서울 3일 여행 계획 세워줘")],
            "query": "서울 3일 여행 계획 세워줘",
            "next_worker": "", "travel_plan": "", "info_result": "", "final_answer": "",
        })
        a = r.get("final_answer") or (r["messages"][-1].content if r.get("messages") else "")
        result("Supervisor pattern", len(a) > 50, f"{len(a)} chars")
    except Exception as e:
        result("Supervisor pattern", False, str(e)[:80])

    # Pattern 2: Sequential
    try:
        from agents.multi_agent import create_sequential_graph
        g = create_sequential_graph()
        r = g.invoke({
            "messages": [HumanMessage(content="부산 여행 가이드를 터키어로 만들어줘")],
            "query": "부산 여행 가이드를 터키어로 만들어줘",
            "target_lang": "tr", "research": "", "plan": "", "translated": "", "final_answer": "",
        })
        a = r.get("translated") or r.get("final_answer") or ""
        result("Sequential pattern", len(a) > 50, f"{len(a)} chars")
    except Exception as e:
        result("Sequential pattern", False, str(e)[:80])

    # Pattern 3: Hierarchical
    try:
        from agents.multi_agent import create_hierarchical_graph
        g = create_hierarchical_graph()
        r = g.invoke({
            "messages": [HumanMessage(content="제주도 여행 계획 + 환율 정보 + 날씨")],
            "query": "제주도 여행 계획 + 환율 정보 + 날씨",
            "travel_done": False, "support_done": False,
            "travel_result": "", "support_result": "", "final_answer": "",
        })
        a = r.get("final_answer") or (r["messages"][-1].content if r.get("messages") else "")
        result("Hierarchical pattern", len(a) > 50, f"{len(a)} chars")
    except Exception as e:
        result("Hierarchical pattern", False, str(e)[:80])


# ──────────────────────────────────────────────
# 3. 검색기 테스트
# ──────────────────────────────────────────────

def test_retrievers():
    section("3. Retriever Comparison Tests")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from retrievers.keyword_retriever import create_keyword_retriever
    from retrievers.vector_retriever import create_vector_retriever
    from retrievers.hybrid_retriever import create_hybrid_retriever
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    # PDF 로드 + 청킹
    data_dir = os.path.join(PROJECT_ROOT, "data")
    documents = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".pdf"):
            documents.extend(PyPDFLoader(os.path.join(data_dir, fn)).load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(documents)
    result("PDF load + chunking", len(chunks) > 0, f"{len(chunks)} chunks")

    query = "서울 맛집 추천"

    # Keyword
    try:
        kr = create_keyword_retriever(chunks)
        kw_res = kr.invoke(query)
        result("Keyword retriever (BM25)", len(kw_res) > 0, f"{len(kw_res)} docs")
    except Exception as e:
        result("Keyword retriever (BM25)", False, str(e)[:80])

    # Vector
    try:
        vr = create_vector_retriever(chunks)
        vec_res = vr.invoke(query)
        result("Vector retriever (Chroma)", len(vec_res) > 0, f"{len(vec_res)} docs")
    except Exception as e:
        result("Vector retriever (Chroma)", False, str(e)[:80])

    # Hybrid
    try:
        hr = create_hybrid_retriever(chunks)
        hyb_res = hr.invoke(query)
        result("Hybrid retriever (Ensemble)", len(hyb_res) > 0, f"{len(hyb_res)} docs")
    except Exception as e:
        result("Hybrid retriever (Ensemble)", False, str(e)[:80])


# ──────────────────────────────────────────────
# 4. FastAPI 엔드포인트 테스트
# ──────────────────────────────────────────────

def test_api():
    section("4. FastAPI Endpoint Tests")

    # uvicorn 시작
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8111"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(5)  # 서버 시작 대기

    base = "http://127.0.0.1:8111"

    try:
        # /health
        try:
            r = requests.get(f"{base}/health", timeout=5)
            data = r.json()
            result("GET /health", data.get("status") == "running" and data.get("model") == "gemma3:4b")
        except Exception as e:
            result("GET /health", False, str(e)[:80])

        # /translate
        try:
            r = requests.post(f"{base}/translate", json={
                "text": "안녕하세요", "source_lang": "ko", "target_lang": "en"
            }, timeout=60)
            data = r.json()
            result("POST /translate", "translated_text" in data and len(data["translated_text"]) > 0, data.get("translated_text", "")[:50])
        except Exception as e:
            result("POST /translate", False, str(e)[:80])

        # /translate error
        try:
            r = requests.post(f"{base}/translate", json={
                "text": "test", "source_lang": "ko", "target_lang": "xx"
            }, timeout=10)
            result("POST /translate (bad lang)", r.status_code == 400)
        except Exception as e:
            result("POST /translate (bad lang)", False, str(e)[:80])

        # /chat
        try:
            r = requests.post(f"{base}/chat", json={
                "message": "서울 날씨 어때?", "session_id": "api-test"
            }, timeout=120)
            data = r.json()
            result("POST /chat", "response" in data and len(data["response"]) > 5, f"{len(data.get('response',''))} chars")
        except Exception as e:
            result("POST /chat", False, str(e)[:80])

        # /chat/stream
        try:
            r = requests.post(f"{base}/chat/stream", json={
                "message": "부산 날씨", "session_id": "stream-test"
            }, timeout=120, stream=True)
            events = []
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                if len(events) > 5:
                    break
            has_start = "start" in events
            has_end_or_token = "token" in events or "end" in events
            result("POST /chat/stream (SSE)", has_start and has_end_or_token, f"events: {events[:5]}")
        except Exception as e:
            result("POST /chat/stream (SSE)", False, str(e)[:80])

        # /tools
        try:
            r = requests.get(f"{base}/tools", timeout=5)
            data = r.json()
            result("GET /tools", len(data.get("tools", [])) == 5, f"{len(data.get('tools',[]))} tools")
        except Exception as e:
            result("GET /tools", False, str(e)[:80])

    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("#  FULL TEST SUITE - Korea Travel AI Assistant")
    print("#" * 65)

    t0 = time.time()

    test_react_agent()
    test_multi_agent()
    test_retrievers()
    test_api()

    elapsed = time.time() - t0

    print(f"\n{'#' * 65}")
    print(f"#  RESULTS: {PASS} passed, {FAIL} failed  ({elapsed:.0f}s)")
    print(f"{'#' * 65}\n")
