"""
RAG 체인 모듈 - 검색 증강 생성(RAG) 체인을 구성합니다.
스트리밍과 대화 기록을 지원합니다.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from config import MODEL_NAME, MAX_TOKENS, SYSTEM_PROMPT, RETRIEVER_K
from loaders.doc_loader import load_vector_store

# 세션별 대화 기록 저장소
_message_histories: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    세션 ID에 해당하는 대화 기록을 반환합니다.
    없으면 새로 생성합니다.

    Args:
        session_id: 세션 식별자

    Returns:
        해당 세션의 대화 기록
    """
    if session_id not in _message_histories:
        _message_histories[session_id] = ChatMessageHistory()
    return _message_histories[session_id]


def format_docs(docs: list) -> str:
    """
    검색된 문서들을 하나의 문자열로 포맷합니다.

    Args:
        docs: 검색된 문서 리스트

    Returns:
        포맷된 컨텍스트 문자열
    """
    if not docs:
        return "검색된 문서가 없습니다."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    """
    RAG 체인을 생성합니다.
    벡터 저장소가 있으면 RAG를 사용하고, 없으면 일반 대화 체인을 생성합니다.

    Returns:
        (chain, is_rag) 튜플 - 체인 객체와 RAG 사용 여부
    """
    # LLM 초기화 (Ollama 로컬 모델, 스트리밍 활성화)
    llm = ChatOllama(
        model=MODEL_NAME,
        num_predict=MAX_TOKENS,
    )

    # 벡터 저장소 로드 시도
    vector_store = load_vector_store()

    if vector_store:
        # RAG 체인 구성
        retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )

        # RAG 프롬프트 템플릿 (SYSTEM_PROMPT에 {context} 포함)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # RAG 체인 구성
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["input"]))
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # 대화 기록이 포함된 체인
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        print("🔍 RAG 모드로 실행합니다.")
        return chain_with_history, True
    else:
        # 일반 대화 체인 (RAG 없음)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()

        # 대화 기록이 포함된 체인
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        print("💬 일반 대화 모드로 실행합니다 (RAG 비활성화).")
        return chain_with_history, False
