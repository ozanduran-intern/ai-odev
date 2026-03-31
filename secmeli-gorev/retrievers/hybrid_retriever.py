"""
하이브리드 검색기 (앙상블)
키워드(BM25) 검색과 벡터(임베딩) 검색을 결합하여 상호 보완적 검색 수행
- 키워드 검색: 정확한 단어 일치에 강함
- 벡터 검색: 의미적 유사성 파악에 강함
- 앙상블: 두 방식의 장점을 결합하여 더 높은 검색 품질 달성
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from config import RETRIEVER_K
from .keyword_retriever import create_keyword_retriever
from .vector_retriever import create_vector_retriever


def create_hybrid_retriever(
    chunks: list[Document],
    k: int = RETRIEVER_K,
    weights: list[float] | None = None,
) -> EnsembleRetriever:
    """
    하이브리드(앙상블) 검색기를 생성하여 반환
    BM25 키워드 검색기와 벡터 검색기를 가중 결합

    Args:
        chunks: 분할된 문서 청크 리스트
        k: 각 검색기가 반환할 문서 수 (기본값: config.RETRIEVER_K)
        weights: [키워드 가중치, 벡터 가중치] (기본값: [0.4, 0.6])

    Returns:
        EnsembleRetriever: 하이브리드 검색기 인스턴스
    """
    if weights is None:
        weights = [0.4, 0.6]  # 벡터 검색에 약간 더 높은 가중치

    # 키워드 검색기 생성 (BM25)
    keyword_retriever = create_keyword_retriever(chunks, k=k)

    # 벡터 검색기 생성 (HuggingFace + Chroma)
    vector_retriever = create_vector_retriever(chunks, k=k)

    # 앙상블 검색기: 두 검색기의 결과를 가중 결합
    ensemble_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        weights=weights,
    )

    return ensemble_retriever
