#선택과제 Beroam — AI Travel Assistant

> Multilingual AI travel assistant powered by a ReAct agent, three multi-agent patterns, hybrid RAG retrieval, and a streaming FastAPI backend.

> 다국어 AI 여행 도우미 — ReAct 에이전트, 3가지 멀티 에이전트 패턴, 하이브리드 RAG 검색, SSE 스트리밍 FastAPI 백엔드 기반.

---

## 1. Project Description / 프로젝트 설명

Beroam is an AI-powered travel assistant that helps users plan trips to any country in the world. It supports Korean, English, Turkish, Japanese, and 10+ other languages — automatically detecting the user's language and responding consistently.

The system uses **Ollama (gemma3:4b)** running locally, meaning no cloud API keys are needed. It combines tool-calling (weather, currency, translation, place recommendations), document retrieval (RAG over Korea travel PDFs), and multi-agent orchestration patterns.

Beroam은 전 세계 어디든 여행 계획을 도와주는 AI 기반 여행 도우미입니다. 한국어, 영어, 터키어, 일본어 등 10개 이상의 언어를 자동 감지하여 일관된 언어로 응답합니다.

시스템은 로컬에서 실행되는 **Ollama (gemma3:4b)**를 사용하므로 클라우드 API 키가 필요 없습니다. 도구 호출(날씨, 환율, 번역, 장소 추천), 문서 검색(한국 여행 PDF 기반 RAG), 멀티 에이전트 오케스트레이션 패턴을 결합합니다.

---

## 2. Features / 주요 기능

| Feature | Description |
|---|---|
| **ReAct Agent** | Reasoning + Acting agent with 5 tools: weather, currency, translation, place recommendation, RAG search |
| **Multi-Agent** | 3 orchestration patterns: Supervisor, Sequential, Hierarchical (LangGraph) |
| **RAG Retrieval** | 3 retriever types: BM25 Keyword, ChromaDB Vector, Hybrid Ensemble |
| **Multilingual** | Auto-detects 15+ languages. Code-level language injection ensures consistent responses |
| **Streaming API** | SSE (Server-Sent Events) for real-time token streaming via FastAPI |
| **Web UI** | Premium Beroam branded UI with hero landing + chat interface |
| **Session Memory** | Per-session chat history with TTL-based automatic cleanup |
| **Global Coverage** | 40+ cities for weather, 25 currencies, 7 cities with 6+ places per category |

---

## 3. Tech Stack / 기술 스택

| Layer | Technology |
|---|---|
| **LLM** | Ollama + gemma3:4b (local, no API key) |
| **Agent Framework** | LangChain (ReAct agent), LangGraph (multi-agent) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (HuggingFace) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Keyword Search** | BM25 via rank-bm25 |
| **API** | FastAPI + SSE (sse-starlette) + Uvicorn |
| **PDF Parsing** | PyPDF |
| **Frontend** | Vanilla HTML/CSS/JS (Beroam branded, dark theme) |
| **Language** | Python 3.14 |

---

## 4. Setup Instructions / 설치 방법

### Prerequisites / 사전 요구사항

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- `gemma3:4b` model pulled: `ollama pull gemma3:4b`

### Installation / 설치

```bash
# Clone and enter directory
cd ai-odev/secmeli-gorev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Run / 실행

```bash
# Start web server (API + UI)
./venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8111

# Open in browser
open http://127.0.0.1:8111

# Or use interactive terminal chat
./venv/bin/python chat.py

# Run full test suite (18 tests)
./venv/bin/python test_all.py
```

---

## 5. Architecture / 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Beroam Web UI                        │
│              (HTML/CSS/JS, SSE Client)                  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Server                        │
│  /chat  /chat/stream  /translate  /health  /tools       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              ReAct Agent (LangChain)                    │
│         Language Detection → Prompt Injection           │
│              Session Memory (TTL=1h)                    │
├─────────┬──────────┬──────────┬──────────┬──────────────┤
│ Weather │ Currency │Translate │ Places   │ RAG Search   │
│  Tool   │  Tool    │  Tool    │  Tool    │  Tool        │
│(40 city)│(25 curr) │(15 lang) │(7+ city) │(Korea PDFs)  │
│         │          │(LLM call)│(DB+LLM)  │              │
└─────────┴──────────┴──────────┴──────────┴──────┬───────┘
                                                  │
                    ┌─────────────────────────────┼───────┐
                    │        RAG Pipeline         │       │
                    │  PDF → Chunks → Embeddings  │       │
                    │                             ▼       │
                    │  ┌─────────┐ ┌──────────┐ ┌─────┐  │
                    │  │Keyword  │ │ Vector   │ │Hybrid│  │
                    │  │(BM25)   │ │(ChromaDB)│ │(Ens.)│  │
                    │  └─────────┘ └──────────┘ └─────┘  │
                    └─────────────────────────────────────┘

Multi-Agent Patterns (LangGraph):
┌───────────────┐ ┌───────────────┐ ┌─────────────────────┐
│  Supervisor   │ │  Sequential   │ │   Hierarchical      │
│  ┌─────────┐  │ │               │ │  ┌──────────────┐   │
│  │Supervisor│  │ │ Research      │ │  │   Manager     │   │
│  └──┬──┬───┘  │ │   ↓           │ │  └──┬───────┬───┘   │
│  ┌──▼┐ ┌▼──┐  │ │ Planner      │ │  ┌──▼──┐ ┌──▼──┐    │
│  │Info│ │Plan│  │ │   ↓           │ │  │Travel│ │Suppt│    │
│  └───┘ └───┘  │ │ Translator    │ │  │Team  │ │Team │    │
│               │ │   ↓           │ │  └─────┘ └─────┘    │
│  → Planner    │ │ Finalizer     │ │                     │
└───────────────┘ └───────────────┘ └─────────────────────┘
```

---

## 6. API Endpoints / API 엔드포인트

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Beroam Web UI (landing page + chat) |
| `GET` | `/health` | Server status + model name |
| `POST` | `/chat` | Full response from ReAct agent |
| `POST` | `/chat/stream` | SSE streaming response (real-time) |
| `POST` | `/translate` | Direct translation (bypasses agent) |
| `GET` | `/tools` | List available tools |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### Request Examples

```bash
# Chat
curl -X POST http://127.0.0.1:8111/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?", "session_id": "demo"}'

# Translate
curl -X POST http://127.0.0.1:8111/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "source_lang": "en", "target_lang": "ko"}'

# Stream (SSE)
curl -N -X POST http://127.0.0.1:8111/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Best food in Paris?", "session_id": "demo"}'
```

---

## 7. Multi-Agent Patterns / 멀티 에이전트 패턴

### Pattern 1: Supervisor (감독자)

A supervisor node decides which worker to call next. Workers execute specialized tasks and return results to the supervisor, who then routes to the next worker or finalizes.

감독자 노드가 다음에 호출할 작업자를 결정합니다. 작업자들은 전문 작업을 수행하고 결과를 감독자에게 반환하며, 감독자가 다음 작업자로 라우팅하거나 최종 결과를 생성합니다.

**Flow:** User → Supervisor → Info Worker → Supervisor → Planner → Supervisor → Final Answer

### Pattern 2: Sequential (순차적)

Fixed pipeline where each node's output feeds the next. Deterministic order, no routing decisions.

각 노드의 출력이 다음 노드의 입력이 되는 고정 파이프라인입니다. 결정론적 순서, 라우팅 결정 없음.

**Flow:** User → Researcher → Planner → Translator → Finalizer → Answer

### Pattern 3: Hierarchical (계층적)

A top-level manager delegates to sub-teams (Travel Team, Support Team), each handling a domain. Results are merged at the top level.

최상위 매니저가 하위 팀(여행 팀, 지원 팀)에게 위임하고, 각 팀이 도메인별로 처리합니다. 결과는 최상위에서 병합됩니다.

**Flow:** User → Manager → [Travel Team | Support Team] → Manager → Final Answer

---

## 8. Retriever Comparison / 검색기 비교

| Retriever | Method | Strength | Weakness |
|---|---|---|---|
| **Keyword (BM25)** | Term frequency matching | Exact keyword matches, fast | Misses synonyms and semantic meaning |
| **Vector (ChromaDB)** | Embedding similarity | Understands meaning, finds related concepts | Can miss exact terms, slower |
| **Hybrid (Ensemble)** | BM25 (40%) + Vector (60%) weighted | Best of both worlds, highest recall | Slightly more compute |

The hybrid retriever uses `EnsembleRetriever` from LangChain, combining BM25 keyword search with ChromaDB vector search at a 0.4:0.6 weight ratio. This ensures both exact term matches and semantic understanding contribute to results.

하이브리드 검색기는 LangChain의 `EnsembleRetriever`를 사용하여 BM25 키워드 검색과 ChromaDB 벡터 검색을 0.4:0.6 가중치로 결합합니다. 이를 통해 정확한 용어 일치와 의미적 이해가 모두 검색 결과에 기여합니다.

---

## 9. Design Decisions / 설계 결정

### Why Ollama + gemma3:4b?
Local execution means zero API costs and no rate limits. gemma3:4b balances capability with speed on consumer hardware (Apple Silicon). No cloud dependency — works offline.

로컬 실행으로 API 비용 제로, 속도 제한 없음. gemma3:4b는 일반 하드웨어(Apple Silicon)에서 성능과 속도를 균형 있게 제공합니다.

### Why ReAct over Function Calling?
gemma3:4b doesn't support native tool-calling APIs. The ReAct (Reasoning + Acting) pattern uses text-based prompting to achieve tool use. A custom `FriendlyReActParser` gracefully handles cases where the model skips formatting — treating unformatted output as a direct conversational response instead of crashing.

gemma3:4b는 네이티브 도구 호출 API를 지원하지 않습니다. ReAct 패턴은 텍스트 기반 프롬프팅으로 도구 사용을 구현합니다. 커스텀 `FriendlyReActParser`가 형식 미준수 출력을 자연스러운 대화 응답으로 처리하여 에러 대신 유연한 응답을 제공합니다.

### Why Code-Level Language Detection?
Small models (4B params) struggle with consistent language matching. Instead of relying on the model to detect the user's language, we detect it programmatically using Unicode ranges and keyword matching, then inject `{reply_language}` directly into the prompt template. This ensures reliable multilingual behavior.

소형 모델(4B 파라미터)은 일관된 언어 매칭이 어렵습니다. 모델에 의존하는 대신 Unicode 범위와 키워드 매칭으로 프로그래밍적으로 언어를 감지하고, `{reply_language}`를 프롬프트 템플릿에 직접 주입합니다.

### Why Simulated Data + LLM Fallback?
Weather and currency use simulated data (no API keys needed for demo). Place recommendations use a hybrid approach: 7 major cities have curated local data (6+ places per category with `random.sample()` for variety), while any other city falls back to LLM-generated recommendations.

날씨와 환율은 시뮬레이션 데이터를 사용합니다(데모용 API 키 불필요). 장소 추천은 하이브리드 방식: 7개 주요 도시는 큐레이팅된 로컬 데이터(카테고리당 6개 이상, `random.sample()`로 다양성 확보), 나머지 도시는 LLM 생성 추천으로 폴백합니다.

### Why SSE over WebSocket?
SSE is simpler, HTTP-native, and sufficient for unidirectional streaming (server → client). No connection upgrade overhead. The chat uses POST requests for user messages and SSE for streaming responses — clean separation of concerns.

SSE는 더 단순하고 HTTP 네이티브이며, 단방향 스트리밍(서버→클라이언트)에 충분합니다. WebSocket의 연결 업그레이드 오버헤드가 없습니다.

### Why TTL-Based Session Cleanup?
In-memory session storage (`_session_histories`) would grow indefinitely. A 1-hour TTL with lazy cleanup on each request prevents memory leaks while keeping active conversations alive. Session history is also capped at 10 messages to prevent small model confusion.

인메모리 세션 저장소가 무한히 증가하는 것을 방지하기 위해 1시간 TTL과 요청마다의 지연 정리를 적용합니다. 세션 기록도 10개 메시지로 제한하여 소형 모델 혼란을 방지합니다.

---

## Test Results / 테스트 결과

```
#################################################################
#  FULL TEST SUITE - Beroam Travel AI Assistant
#################################################################

1. ReAct Agent Tests
  [PASS] Weather single tool
  [PASS] Currency single tool
  [PASS] Multi tool (weather+food)
  [PASS] Translate Turkish->Korean
  [PASS] RAG search Busan

2. Multi-Agent Pattern Tests
  [PASS] Supervisor pattern
  [PASS] Sequential pattern
  [PASS] Hierarchical pattern

3. Retriever Comparison Tests
  [PASS] PDF load + chunking (13 chunks)
  [PASS] Keyword retriever (BM25) — 3 docs
  [PASS] Vector retriever (Chroma) — 3 docs
  [PASS] Hybrid retriever (Ensemble) — 4 docs

4. FastAPI Endpoint Tests
  [PASS] GET /health
  [PASS] POST /translate
  [PASS] POST /translate (bad lang → 400)
  [PASS] POST /chat
  [PASS] POST /chat/stream (SSE)
  [PASS] GET /tools

RESULTS: 18 passed, 0 failed
#################################################################
```

---

## Project Structure / 프로젝트 구조

```
secmeli-gorev/
├── config.py                 # Model, embedding, chunking settings
├── chat.py                   # Interactive terminal chat
├── test_all.py               # Full test suite (18 tests)
├── test_retrievers.py        # Retriever-specific tests
├── requirements.txt          # Python dependencies
├── agents/
│   ├── react_agent.py        # ReAct agent + language detection + custom parser
│   └── multi_agent.py        # 3 multi-agent patterns (LangGraph)
├── tools/
│   ├── weather.py            # Weather lookup (40+ cities)
│   ├── currency.py           # Currency conversion (25 currencies)
│   ├── translator.py         # LLM-based translation (15 languages)
│   └── place_recommender.py  # Place recommendations (DB + LLM fallback)
├── retrievers/
│   ├── keyword_retriever.py  # BM25 keyword search
│   ├── vector_retriever.py   # ChromaDB vector search
│   └── hybrid_retriever.py   # Ensemble (BM25 + Vector)
├── api/
│   ├── main.py               # FastAPI server (REST + SSE + CORS)
│   └── static/
│       └── index.html        # Beroam Web UI
├── data/
│   ├── korea_travel_guide.pdf
│   ├── korea_food_guide.pdf
│   └── korea_transport_guide.pdf
└── chroma_db/                # Persistent vector store
```
