#AI 과제
0.공통 과제

AI-related assignments and projects.

AI 관련 과제 및 프로젝트 모음입니다.

---

## Project Structure / 프로젝트 구조

```
ai-odev/
├── ortak-gorev/        # Korean Learning Chatbot / 한국어 학습 챗봇
│   ├── app.py          # Main entry point / 메인 진입점
│   ├── config.py       # Configuration / 설정
│   ├── chains/         # LangChain RAG chains / RAG 체인
│   ├── loaders/        # Document loaders / 문서 로더
│   ├── data/           # Korean learning PDFs / 한국어 학습 PDF
│   └── README.md       # Detailed docs / 상세 문서
└── README.md           # This file / 이 파일
```

## Projects / 프로젝트

### ortak-gorev — Korean Learning Chatbot / 한국어 학습 챗봇

A multilingual Korean language learning chatbot with:

- **Local LLM** — Ollama + gemma3:4b (no API key needed)
- **RAG** — Retrieves context from Korean grammar, vocabulary, and expression PDFs
- **Streaming** — Real-time token-by-token responses
- **Multi-turn memory** — Remembers conversation context within a session
- **Trilingual** — Korean + English + Turkish translations

See [ortak-gorev/README.md](ortak-gorev/README.md) for setup instructions and design decisions.

자세한 내용은 [ortak-gorev/README.md](ortak-gorev/README.md)를 참고하세요.
