"""
Interactive Chat - Korea Travel AI Assistant
터미널에서 LLM처럼 대화할 수 있는 인터페이스 (깔끔한 UI)
"""

import os
import sys
import io
import itertools
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ── 색상 코드 ──
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"


def print_banner():
    print(f"""
{CYAN}{BOLD}╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   {WHITE}Korea Travel AI Assistant{CYAN}                          ║
║   {DIM}{WHITE}한국 여행 AI 어시스턴트{RESET}{CYAN}{BOLD}                            ║
║                                                       ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║   {YELLOW}날씨{CYAN}  서울 날씨 어때?                              ║
║   {YELLOW}환율{CYAN}  100 USD 몇 원이야?                          ║
║   {YELLOW}번역{CYAN}  Merhaba를 한국어로                          ║
║   {YELLOW}추천{CYAN}  부산 맛집 추천해줘                           ║
║   {YELLOW}검색{CYAN}  What to do in Jeju?                        ║
║                                                       ║
║   {DIM}종료: quit | exit | q{RESET}{CYAN}{BOLD}                            ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝{RESET}
""")


def spinner(stop_event):
    """답변 대기 중 애니메이션"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for frame in itertools.cycle(frames):
        if stop_event.is_set():
            break
        print(f"\r  {DIM}{frame} 생각하는 중...{RESET}", end="", flush=True)
        time.sleep(0.1)
    print("\r" + " " * 30 + "\r", end="", flush=True)


def format_tool_steps(steps):
    """중간 도구 호출을 한 줄 요약으로 포맷"""
    if not steps:
        return ""
    tools_used = []
    for action, _ in steps:
        name = action.tool
        icons = {
            "get_weather": "🌤",
            "convert_currency": "💱",
            "translate": "🌐",
            "recommend_places": "📍",
            "search_travel_docs": "📚",
        }
        icon = icons.get(name, "🔧")
        tools_used.append(f"{icon} {name}")
    return f"  {DIM}사용된 도구: {' → '.join(tools_used)}{RESET}\n"


def main():
    print_banner()

    # verbose=False로 에이전트 로딩 (내부 출력 숨기기)
    print(f"  {DIM}에이전트 로딩 중...{RESET}", end="", flush=True)

    # 에이전트 생성 시 stdout 억제
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from agents.react_agent import create_agent
        agent = create_agent()
    finally:
        sys.stdout = old_stdout

    print(f"\r  {GREEN}준비 완료!{RESET}              \n")

    config = {"configurable": {"session_id": "interactive"}}

    while True:
        try:
            user_input = input(f"  {GREEN}{BOLD}You >{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {CYAN}안녕히 가세요! 좋은 여행 되세요! ✈{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n  {CYAN}안녕히 가세요! 좋은 여행 되세요! ✈{RESET}\n")
            break

        # 스피너 시작
        stop = threading.Event()
        spin_thread = threading.Thread(target=spinner, args=(stop,), daemon=True)
        spin_thread.start()

        try:
            # agent verbose 출력 억제
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                result = agent.invoke({"input": user_input}, config=config)
            finally:
                sys.stdout = old_stdout

            stop.set()
            spin_thread.join()

            # 도구 사용 요약
            steps = result.get("intermediate_steps", [])
            tool_summary = format_tool_steps(steps)

            # 응답 출력
            output = result.get("output", "")
            print(f"\n  {CYAN}{BOLD}AI >{RESET} {output}\n")
            if tool_summary:
                print(tool_summary)
            print(f"  {DIM}{'─' * 50}{RESET}\n")

        except Exception as e:
            stop.set()
            spin_thread.join()
            sys.stdout = old_stdout
            print(f"\n  {MAGENTA}오류: {e}{RESET}\n")


if __name__ == "__main__":
    main()
