"""
문서 로더 모듈 - 다양한 형식의 문서를 로드하고 벡터 저장소를 생성합니다.
"""

import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_DB_PATH


def get_embeddings():
    """임베딩 모델을 초기화하고 반환합니다."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents(data_dir: str = "./data") -> list:
    """
    데이터 디렉토리에서 모든 문서를 로드합니다.

    Args:
        data_dir: 문서가 저장된 디렉토리 경로

    Returns:
        로드된 문서 리스트
    """
    documents = []

    if not os.path.exists(data_dir):
        print(f"⚠️ 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return documents

    # 텍스트 파일 로드
    txt_loader = DirectoryLoader(
        data_dir, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    documents.extend(txt_loader.load())

    # PDF 파일 로드
    pdf_loader = DirectoryLoader(
        data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    documents.extend(pdf_loader.load())

    print(f"📄 총 {len(documents)}개의 문서를 로드했습니다.")
    return documents


def split_documents(documents: list) -> list:
    """
    문서를 청크 단위로 분할합니다.

    Args:
        documents: 분할할 문서 리스트

    Returns:
        분할된 청크 리스트
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ {len(chunks)}개의 청크로 분할했습니다.")
    return chunks


def create_vector_store(data_dir: str = "./data") -> Chroma | None:
    """
    문서를 로드하고 벡터 저장소를 생성합니다.

    Args:
        data_dir: 문서가 저장된 디렉토리 경로

    Returns:
        Chroma 벡터 저장소 또는 문서가 없으면 None
    """
    documents = load_documents(data_dir)

    if not documents:
        print("📭 로드할 문서가 없습니다. RAG 없이 실행됩니다.")
        return None

    chunks = split_documents(documents)
    embeddings = get_embeddings()

    # 벡터 저장소 생성 및 저장
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    print(f"💾 벡터 저장소가 생성되었습니다: {VECTOR_DB_PATH}")
    return vector_store


def load_vector_store() -> Chroma | None:
    """
    기존 벡터 저장소를 로드합니다. 없으면 새로 생성합니다.

    Returns:
        Chroma 벡터 저장소 또는 None
    """
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_DB_PATH):
        print("📂 기존 벡터 저장소를 로드합니다.")
        return Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
        )

    return create_vector_store()
