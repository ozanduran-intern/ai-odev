# 프로젝트 설정 파일
# Ollama 기반 로컬 LLM 및 임베딩 설정

# 사용할 LLM 모델명
MODEL_NAME = "gemma3:4b"

# Ollama 서버 기본 URL
BASE_URL = "http://localhost:11434"

# 문서 청킹 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 임베딩 모델
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 벡터 데이터베이스 경로
VECTOR_DB_PATH = "./chroma_db"

# 검색 시 반환할 문서 수
RETRIEVER_K = 3
