"""챗봇 테스트 스크립트 - 스트리밍, RAG, 멀티턴 기능을 검증합니다."""

import sys
import time
from chains.rag_chain import create_rag_chain

# 테스트 질문 목록
QUESTIONS = [
    "How do I say hello in Korean?",
    "What are the basic Korean particles?",
    "Tell me about Korean honorific system",
    "이/가 and 은/는 차이가 뭐야?",
    "Merhaba Korece'de nasıl denir?",
    # 멀티턴 테스트: 이전 대화를 기억하는지 확인
    "Can you summarize what we talked about so far?",
]

def test_chatbot():
    print("=" * 70)
    print("🧪 챗봇 테스트 시작")
    print("=" * 70)

    # 체인 생성
    chain, is_rag = create_rag_chain()
    print(f"\n📌 RAG mode: {is_rag}\n")

    session_id = "test_session"
    config = {"configurable": {"session_id": session_id}}

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'─' * 70}")
        print(f"🧑 Question {i}: {question}")
        print(f"{'─' * 70}")
        print("🤖 선생님: ", end="", flush=True)

        token_count = 0
        start = time.time()

        try:
            for chunk in chain.stream({"input": question}, config=config):
                print(chunk, end="", flush=True)
                token_count += 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

        elapsed = time.time() - start
        print(f"\n\n  ⏱️ {elapsed:.1f}s | 📊 {token_count} chunks streamed")

    print(f"\n{'=' * 70}")
    print("✅ 모든 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    test_chatbot()
