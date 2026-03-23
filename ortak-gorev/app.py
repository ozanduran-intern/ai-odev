"""
메인 애플리케이션 - 한국어 학습 챗봇의 진입점입니다.
스트리밍 방식으로 토큰 단위 응답을 제공합니다.
"""

from chains.rag_chain import create_rag_chain


def print_welcome():
    """환영 메시지를 출력합니다."""
    print("\n" + "=" * 60)
    print("🇰🇷  한국어 학습 챗봇  /  Korean Learning Chatbot")
    print("=" * 60)
    print("한국어를 배워봅시다! 질문을 입력하세요.")
    print("Let's learn Korean! Type your question.")
    print("Korece öğrenelim! Sorunuzu yazın.")
    print("-" * 60)
    print("종료: 'quit' 또는 'exit' 입력")
    print("=" * 60 + "\n")


def main():
    """챗봇의 메인 루프를 실행합니다."""
    print_welcome()

    # RAG 체인 생성
    chain, is_rag = create_rag_chain()

    # 세션 ID 설정
    session_id = "default_session"
    config = {"configurable": {"session_id": session_id}}

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 안녕히 가세요! Goodbye! Hoşça kalın!")
            break

        # 종료 명령 확인
        if user_input.lower() in ("quit", "exit", "종료"):
            print("\n👋 안녕히 가세요! Goodbye! Hoşça kalın!")
            break

        # 빈 입력 무시
        if not user_input:
            continue

        # 스트리밍 응답 출력
        print("\n🤖 선생님: ", end="", flush=True)
        try:
            for chunk in chain.stream(
                {"input": user_input},
                config=config,
            ):
                print(chunk, end="", flush=True)
            print()  # 줄바꿈
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
