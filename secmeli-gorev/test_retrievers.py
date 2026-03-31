"""
검색기 테스트 스크립트
3가지 검색기(키워드, 벡터, 하이브리드)를 생성하고 동일한 쿼리로 비교 테스트
"""

import os
import sys
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 프로젝트 루트 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH
from retrievers.keyword_retriever import create_keyword_retriever
from retrievers.vector_retriever import create_vector_retriever
from retrievers.hybrid_retriever import create_hybrid_retriever


def load_and_chunk_pdfs() -> list:
    """data/ 폴더의 PDF 문서를 로드하고 청크로 분할"""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    documents = []

    print("[1/4] PDF 문서 로드 중...")
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"  - {filename}: {len(docs)}페이지 로드")

    print(f"  총 {len(documents)}페이지 로드 완료\n")

    # 청킹
    print("[2/4] 문서 청킹 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  청크 크기: {CHUNK_SIZE}, 오버랩: {CHUNK_OVERLAP}")
    print(f"  총 {len(chunks)}개 청크 생성 완료\n")

    return chunks


def print_results(name: str, docs: list, elapsed: float):
    """검색 결과를 보기 좋게 출력"""
    print(f"\n{'─' * 60}")
    print(f"  {name} 결과 ({len(docs)}건, {elapsed:.2f}초)")
    print(f"{'─' * 60}")
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "?"))
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        # 미리보기: 첫 150자
        preview = content[:150].replace("\n", " ")
        if len(content) > 150:
            preview += "..."
        print(f"  [{i}] (출처: {source}, p.{page})")
        print(f"      {preview}")
        print()


def main():
    print("=" * 60)
    print("  검색기 비교 테스트 (키워드 vs 벡터 vs 하이브리드)")
    print("=" * 60)
    print()

    # 1) PDF 로드 + 청킹
    chunks = load_and_chunk_pdfs()

    # 기존 chroma_db 삭제 (깨끗한 테스트를 위해)
    db_path = os.path.join(PROJECT_ROOT, VECTOR_DB_PATH)

    # 2) 검색기 생성
    print("[3/4] 검색기 생성 중...")

    # 키워드 검색기 (BM25)
    t0 = time.time()
    keyword_retriever = create_keyword_retriever(chunks)
    t_kw = time.time() - t0
    print(f"  BM25 키워드 검색기 생성 완료 ({t_kw:.2f}초)")

    # 벡터 검색기 (HuggingFace + Chroma)
    t0 = time.time()
    vector_retriever = create_vector_retriever(chunks)
    t_vec = time.time() - t0
    print(f"  벡터 검색기 생성 완료 ({t_vec:.2f}초)")

    # 하이브리드 검색기 (앙상블)
    t0 = time.time()
    hybrid_retriever = create_hybrid_retriever(chunks)
    t_hyb = time.time() - t0
    print(f"  하이브리드 검색기 생성 완료 ({t_hyb:.2f}초)")
    print()

    # 3) 검색 테스트
    query = "서울 맛집 추천"
    print(f"[4/4] 검색 쿼리: \"{query}\"")

    # 키워드 검색
    t0 = time.time()
    kw_results = keyword_retriever.invoke(query)
    t_kw_search = time.time() - t0
    print_results("키워드 검색 (BM25)", kw_results, t_kw_search)

    # 벡터 검색
    t0 = time.time()
    vec_results = vector_retriever.invoke(query)
    t_vec_search = time.time() - t0
    print_results("벡터 검색 (Semantic)", vec_results, t_vec_search)

    # 하이브리드 검색
    t0 = time.time()
    hyb_results = hybrid_retriever.invoke(query)
    t_hyb_search = time.time() - t0
    print_results("하이브리드 검색 (BM25 0.4 + Vector 0.6)", hyb_results, t_hyb_search)

    # 4) 비교 요약
    print(f"\n{'═' * 60}")
    print("  비교 요약")
    print(f"{'═' * 60}")
    print(f"  {'검색기':<30} {'생성 시간':>10} {'검색 시간':>10} {'결과 수':>8}")
    print(f"  {'─' * 58}")
    print(f"  {'키워드 (BM25)':<30} {t_kw:>9.2f}s {t_kw_search:>9.3f}s {len(kw_results):>8}")
    print(f"  {'벡터 (Semantic)':<30} {t_vec:>9.2f}s {t_vec_search:>9.3f}s {len(vec_results):>8}")
    print(f"  {'하이브리드 (Ensemble)':<30} {t_hyb:>9.2f}s {t_hyb_search:>9.3f}s {len(hyb_results):>8}")

    # 출처 비교
    print(f"\n  출처 비교:")
    for name, results in [("키워드", kw_results), ("벡터", vec_results), ("하이브리드", hyb_results)]:
        sources = [f"{os.path.basename(d.metadata.get('source', '?'))}:p{d.metadata.get('page', '?')}" for d in results]
        print(f"  {name:<12}: {', '.join(sources)}")

    print(f"\n{'═' * 60}")
    print("  테스트 완료")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
