# Korean Learning Chatbot / 한국어 학습 챗봇

A multilingual Korean language learning chatbot powered by a local LLM (Ollama + Gemma 3) with RAG (Retrieval-Augmented Generation), streaming responses, and multi-turn conversation memory.

로컬 LLM(Ollama + Gemma 3)을 활용한 다국어 한국어 학습 챗봇입니다. RAG(검색 증강 생성), 스트리밍 응답, 멀티턴 대화 기록을 지원합니다.

---

## Features / 기능

- **Streaming Responses / 스트리밍 응답** — Token-by-token output for real-time interaction
- **Multi-turn Conversation / 다중 턴 대화** — Remembers chat history within a session using `RunnableWithMessageHistory`
- **RAG / 검색 증강 생성** — Retrieves relevant context from Korean language PDF documents
- **Trilingual Output / 3개 국어 출력** — Korean words with English & Turkish translations (no romanization)
- **Local Inference / 로컬 추론** — Runs entirely on your machine via Ollama, no API key required
- **Modular Architecture / 모듈형 구조** — Clean separation of config, loaders, and chains

---

## Tech Stack / 기술 스택

| Component | Technology |
|-----------|------------|
| LLM | [Ollama](https://ollama.com) + `gemma3:4b` (local) |
| Framework | [LangChain](https://python.langchain.com) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector Store | [Chroma](https://www.trychroma.com) (local) |
| Streaming | LangChain `chain.stream()` |
| Chat History | `RunnableWithMessageHistory` |
| Document Loading | LangChain `DirectoryLoader` (PDF, TXT) |

---

## Architecture / 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
│                    (Korean/English/Turkish)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      app.py (진입점)                          │
│                  Interactive chat loop                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                 chains/rag_chain.py                           │
│  ┌─────────────────────────────────────────────────────┐     │
│  │            RunnableWithMessageHistory                │     │
│  │  ┌───────────────────────────────────────────────┐  │     │
│  │  │  RunnablePassthrough.assign(context=retriever) │  │     │
│  │  │         ▼                                      │  │     │
│  │  │  ChatPromptTemplate (system + history + input) │  │     │
│  │  │         ▼                                      │  │     │
│  │  │  ChatOllama (gemma3:4b, streaming)             │  │     │
│  │  │         ▼                                      │  │     │
│  │  │  StrOutputParser                               │  │     │
│  │  └───────────────────────────────────────────────┘  │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────┬──────────────────────────────────┬────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌─────────────────────────────┐
│   Ollama (Local)    │          │  loaders/doc_loader.py      │
│   gemma3:4b model   │          │  ┌───────────────────────┐  │
│   localhost:11434   │          │  │ DirectoryLoader (PDF) │  │
└─────────────────────┘          │  │         ▼             │  │
                                 │  │ RecursiveCharacter    │  │
                                 │  │ TextSplitter          │  │
                                 │  │         ▼             │  │
                                 │  │ HuggingFace Embeddings│  │
                                 │  │         ▼             │  │
                                 │  │ Chroma Vector Store   │  │
                                 │  └───────────────────────┘  │
                                 └─────────────────────────────┘
```

---

## Project Structure / 프로젝트 구조

```
ortak-gorev/
├── app.py                  # 메인 진입점 (대화형 루프, 스트리밍 출력)
├── config.py               # 모든 설정값 관리 (모델, 청크, 프롬프트)
├── chains/
│   ├── __init__.py
│   └── rag_chain.py        # RAG 체인 구성 (LLM + Retriever + History)
├── loaders/
│   ├── __init__.py
│   └── doc_loader.py       # 문서 로드, 분할, 벡터 저장소 생성
├── data/                   # 한국어 학습 PDF 문서
│   ├── korean_grammar.pdf
│   ├── korean_vocabulary.pdf
│   └── korean_expressions.pdf
├── .env                    # 환경 변수 (커밋 제외)
├── .gitignore
├── requirements.txt
├── test_chatbot.py         # 자동 테스트 스크립트
├── generate_pdfs.py        # PDF 학습 자료 생성 스크립트
└── README.md
```

---

## Setup / 설치

### Prerequisites / 사전 요구사항

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

### Installation / 설치 방법

```bash
# 1. Ollama 설치 및 모델 다운로드
brew install ollama
brew services start ollama
ollama pull gemma3:4b

# 2. 가상 환경 생성 및 활성화
cd ai-odev/ortak-gorev
python3 -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 실행
python app.py
```

### Adding Custom Documents / 커스텀 문서 추가

Place `.txt` or `.pdf` files in the `data/` directory. The chatbot indexes them automatically on startup.

`data/` 디렉토리에 `.txt` 또는 `.pdf` 파일을 넣으면 시작 시 자동으로 인덱싱됩니다.

To re-index after adding new documents, delete the `chroma_db/` directory and restart.

새 문서 추가 후 재인덱싱하려면 `chroma_db/` 디렉토리를 삭제하고 다시 시작하세요.

---

## Usage / 사용법

```
🧑 You: How do I say hello in Korean?
🤖 선생님: "Hello" in Korean is 안녕하세요 (formal) or 안녕 (informal/casual).

안녕하세요 is used in most situations - with strangers, at work, with elders.
안녕 is for close friends and people younger than you.

Example: 선생님, 안녕하세요! - Hello, teacher! / Merhaba, öğretmen!

🧑 You: What about the informal version?
🤖 선생님: The informal version is 안녕.
(Multi-turn memory: the bot remembers we were talking about "hello")
```

The chatbot gives concise, direct answers without tables or excessive formatting. It remembers previous context within a session — follow-up questions reference earlier topics automatically.

챗봇은 테이블이나 과도한 포맷 없이 간결하고 직접적인 답변을 제공합니다. 세션 내에서 이전 대화를 기억하며, 후속 질문은 자동으로 이전 주제를 참조합니다.

---

## Test Results / 테스트 결과

### Streaming / 스트리밍

- Token-by-token output verified / 토큰 단위 출력 검증 완료
- Response chunks: 9-16 per answer (concise responses) / 응답 청크: 답변당 9-16개 (간결한 응답)
- Local inference time: ~3-10 seconds per response / 로컬 추론 시간: 응답당 약 3-10초

### Multi-turn Memory / 멀티턴 메모리

| Turn | Question | Context Retention |
|------|----------|-------------------|
| 1 | "How do I say hello in Korean?" | Initial topic introduced / 초기 주제 도입 |
| 2 | "What about the informal version?" | Remembered "hello" context without explicit mention / "hello"를 언급하지 않았지만 맥락 유지 |
| 3 | "How do I use it in a sentence?" | Remembered 안녕 from turn 2, provided example sentences / 2턴의 안녕을 기억하고 예문 제공 |

### RAG / 검색 증강 생성

- 3 PDF documents loaded (grammar, vocabulary, expressions) / 3개 PDF 문서 로드 (문법, 어휘, 표현)
- 19 pages → 26 chunks via RecursiveCharacterTextSplitter / 19페이지 → 26개 청크로 분할
- Vector store: Chroma with HuggingFace embeddings / 벡터 저장소: Chroma + HuggingFace 임베딩
- Retriever returns top 3 relevant chunks per query / 질의당 상위 3개 관련 청크 반환

---

## Design Decisions / 설계 결정

### Why Ollama (Local LLM) / Ollama를 선택한 이유

Ollama enables fully local inference with no API key, no usage costs, and no data leaving your machine. This makes the project accessible to anyone and eliminates dependency on external services. The `gemma3:4b` model provides a good balance between quality and resource usage on consumer hardware.

Ollama는 API 키 없이 완전한 로컬 추론을 가능하게 합니다. 사용 비용이 없고 데이터가 외부로 나가지 않습니다. `gemma3:4b` 모델은 일반 하드웨어에서 품질과 리소스 사용의 균형을 제공합니다.

### Why RecursiveCharacterTextSplitter / RecursiveCharacterTextSplitter를 선택한 이유

This splitter tries multiple separators (`\n\n`, `\n`, `.`, ` `) in order, preserving semantic coherence within chunks. Unlike a simple character splitter, it avoids breaking sentences mid-word or splitting paragraphs in unnatural places. This is critical for Korean text where sentence boundaries carry meaning.

이 분할기는 여러 구분자(`\n\n`, `\n`, `.`, ` `)를 순서대로 시도하여 청크 내 의미적 일관성을 유지합니다. 단순 문자 분할과 달리, 단어 중간이나 비자연스러운 위치에서 문단이 잘리는 것을 방지합니다.

### Why Chroma / Chroma를 선택한 이유

Chroma is a lightweight, local-first vector database that requires no external server. It stores embeddings on disk, supports persistent storage, and integrates seamlessly with LangChain. For a prototyping and learning project, it provides the simplest path to a working RAG system without infrastructure overhead.

Chroma는 외부 서버가 필요 없는 경량 로컬 벡터 데이터베이스입니다. 임베딩을 디스크에 저장하고 LangChain과 원활하게 통합됩니다. 프로토타이핑 프로젝트에 가장 간단한 RAG 구현 경로를 제공합니다.

### Why chunk_size=1000, overlap=200 / 청크 크기 설정 이유

- **1000 characters**: Large enough to contain a complete concept (e.g., a grammar rule with examples) but small enough to be relevant when retrieved. Korean text is denser than English — 1000 chars covers roughly 2-3 paragraphs.
- **200 character overlap (20%)**: Ensures context isn't lost at chunk boundaries. If a sentence spans two chunks, the overlap captures it in both, improving retrieval accuracy.

- **1000자**: 하나의 완전한 개념(예: 문법 규칙과 예문)을 포함할 만큼 크지만, 검색 시 관련성을 유지할 만큼 작습니다.
- **200자 오버랩 (20%)**: 청크 경계에서 컨텍스트가 손실되지 않도록 합니다.

### Why HuggingFace Embeddings / HuggingFace 임베딩을 선택한 이유

`sentence-transformers/all-MiniLM-L6-v2` runs entirely locally, is free, and produces high-quality 384-dimensional embeddings. It supports multilingual text, which is essential for a project mixing Korean, English, and Turkish. No API key or network connection required for embedding generation.

`all-MiniLM-L6-v2`는 완전히 로컬에서 실행되며 무료입니다. 384차원의 고품질 임베딩을 생성하고 다국어 텍스트를 지원하여 한국어, 영어, 터키어가 혼합된 프로젝트에 적합합니다.

### Why Modular Structure / 모듈형 구조를 선택한 이유

Each module has a single responsibility:
- `config.py` — All settings in one place, easy to change models or parameters
- `loaders/` — Document ingestion, independent of the LLM choice
- `chains/` — LLM chain composition, independent of document loading
- `app.py` — UI/interaction only, delegates all logic to other modules

This separation means you can swap the LLM (Ollama → Claude → Gemini), change the vector store (Chroma → FAISS), or modify the document pipeline — all without touching other modules.

각 모듈은 단일 책임을 가집니다. 이 분리를 통해 LLM, 벡터 저장소, 문서 파이프라인을 다른 모듈을 수정하지 않고 교체할 수 있습니다.

---

## Future Improvements / 향후 개선 사항

- **Agent-based architecture / 에이전트 기반 구조**: Add ReAct agent with tools for translation, grammar checking, and vocabulary quizzes / 번역, 문법 검사, 어휘 퀴즈 도구를 갖춘 ReAct 에이전트 추가
- **More document sources / 추가 문서 소스**: TOPIK exam materials, Korean drama scripts, news articles / TOPIK 시험 자료, 한국 드라마 대본, 뉴스 기사
- **Persistent chat history / 영구 대화 기록**: Store conversations in a database instead of in-memory / 메모리 대신 데이터베이스에 대화 저장
- **Web UI / 웹 인터페이스**: FastAPI + frontend for browser-based interaction / FastAPI + 프론트엔드로 브라우저 기반 상호작용
- **Larger model support / 더 큰 모델 지원**: Option to use gemma3:12b or cloud APIs for higher quality responses / gemma3:12b 또는 클라우드 API를 사용한 고품질 응답 옵션
- **Evaluation metrics / 평가 지표**: Add RAGAS or similar framework to measure RAG quality / RAGAS 등의 프레임워크로 RAG 품질 측정
