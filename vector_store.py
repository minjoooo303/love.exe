
import os
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---- 설정 ----
_embeddings = None
PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "data/faiss_index")  # 디스크 저장 경로

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        # 코사인 유사도 스케일 안정화를 위해 정규화 권장
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def initialize_vector_store():
    """
    디스크에서 FAISS 인덱스를 로드하고(있으면),
    없으면 더미로 새로 만든 뒤 저장.
    """
    emb = _get_embeddings()
    if os.path.exists(PERSIST_DIR):
        vs = FAISS.load_local(PERSIST_DIR, emb, allow_dangerous_deserialization=True)
        print(f"FAISS 로드 완료 → {PERSIST_DIR}")
    else:
        _ensure_dir(PERSIST_DIR)
        vs = FAISS.from_texts(
            ["__DUMMY__INITIAL__ENTRY__"],
            emb,
            distance_strategy=DistanceStrategy.COSINE,  # 코사인 고정
            metadatas=[{"is_dummy": True}]
        )
        vs.save_local(PERSIST_DIR)
        print(f"FAISS 초기화(더미 포함) 및 저장 → {PERSIST_DIR}")
    return vs

def _remove_dummy_if_exists(vector_store: FAISS):
    """더미 문서가 있으면 제거 (최초 1회만 필요)"""
    try:
        # FAISS 내부 docstore 접근 (LangChain 구조)
        keys = list(vector_store.docstore._dict.keys())
        to_delete = [k for k in keys
                     if getattr(vector_store.docstore._dict[k], "metadata", {}).get("is_dummy")]
        if to_delete:
            for k in to_delete:
                del vector_store.docstore._dict[k]
            # 인덱스에서도 벡터 삭제
            # 주의: LangChain의 FAISS는 내부적으로 index와 docstore를 따로 가집니다.
            # 안전하게는 인덱스를 재구축하는 편이 확실합니다.
            # 간단한 방법: 남은 문서로 재생성
            remaining_docs = list(vector_store.docstore._dict.values())
            texts = [d.page_content for d in remaining_docs]
            metas = [d.metadata for d in remaining_docs]
            emb = _get_embeddings()
            new_vs = FAISS.from_texts(texts, emb,
                                      distance_strategy=DistanceStrategy.COSINE,
                                      metadatas=metas)
            return new_vs
    except Exception:
        pass
    return vector_store

def save_vector_store(vector_store: FAISS):
    _ensure_dir(PERSIST_DIR)
    vector_store.save_local(PERSIST_DIR)
    print(f"FAISS 저장 → {PERSIST_DIR}")

def add_story_to_vector_store(vector_store: FAISS, story_content: str, story_id: str, persist: bool = True):
    """
    사연을 벡터 스토어에 추가하고, persist=True면 즉시 디스크에도 저장.
    """
    document = Document(page_content=story_content, metadata={"story_id": story_id})
    vector_store.add_documents([document])

    new_vs = _remove_dummy_if_exists(vector_store)

    if new_vs is not vector_store:
        vector_store = new_vs  # 호출측이 참조를 유지한다면 반환값으로 돌려주는 것도 방법

    if persist:
        save_vector_store(vector_store)
    print(f"사연 (ID: {story_id})이 벡터 스토어에 추가되었습니다. (persist={persist})")

def get_retriever(vector_store: FAISS, k: int = 4, score_threshold: float = 0.7):
    """
    유사도 임계값 기반 리트리버 반환.
    """
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )
