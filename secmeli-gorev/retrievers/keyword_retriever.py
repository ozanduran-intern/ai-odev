"""
키워드 기반 검색기 (BM25)
BM25 알고리즘을 사용한 전통적인 키워드 매칭 검색
문서의 단어 빈도(TF)와 역문서 빈도(IDF)를 활용하여 관련 문서를 찾음
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import RETRIEVER_K


def create_keyword_retriever(chunks: list[Document], k: int = RETRIEVER_K) -> BM25Retriever:
    """
    BM25 키워드 검색기를 생성하여 반환

    Args:
        chunks: 분할된 문서 청크 리스트
        k: 검색 시 반환할 문서 수 (기본값: config.RETRIEVER_K)

    Returns:
        BM25Retriever: 키워드 기반 검색기 인스턴스
    """
    # BM25 검색기 생성 - 문서의 단어 빈도 기반 인덱스를 자동으로 구축
    retriever = BM25Retriever.from_documents(chunks, k=k)

    return retriever
