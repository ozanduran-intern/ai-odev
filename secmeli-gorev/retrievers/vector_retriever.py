"""
벡터 기반 검색기
HuggingFace 임베딩 + ChromaDB를 사용한 의미론적(semantic) 유사도 검색
단어 일치가 아닌 의미적 유사성으로 관련 문서를 찾음
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config import EMBEDDING_MODEL, VECTOR_DB_PATH, RETRIEVER_K


def create_vector_retriever(
    chunks: list[Document],
    k: int = RETRIEVER_K,
    persist_directory: str = VECTOR_DB_PATH,
) -> VectorStoreRetriever:
    """
    벡터 검색기를 생성하여 반환

    Args:
        chunks: 분할된 문서 청크 리스트
        k: 검색 시 반환할 문서 수 (기본값: config.RETRIEVER_K)
        persist_directory: ChromaDB 저장 경로 (기본값: config.VECTOR_DB_PATH)

    Returns:
        VectorStoreRetriever: 벡터 기반 검색기 인스턴스
    """
    # HuggingFace 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ChromaDB 벡터 스토어 생성 (문서 임베딩 후 저장)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # 벡터 스토어를 LangChain Retriever 인터페이스로 변환
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k},
    )

    return retriever
