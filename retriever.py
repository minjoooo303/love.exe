import asyncio
from typing import List, Optional, Tuple
from langchain_core.documents import Document


class ThresholdWrapperRetriever:
    """
    Base retriever에서 문서를 넉넉히 받아온 뒤,
    (FAISS + COSINE 가정) 거리를 relevance로 변환하고 threshold/k로 필터링.
    """
    def __init__(self, base_retriever, vector_store, k: int = 4, score_threshold: Optional[float] = 0.7, prefetch_factor: int = 2):
        self.base_retriever = base_retriever
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold
        self.prefetch = max(k, k * prefetch_factor)

    @staticmethod
    def _cosine_distance_to_relevance(distance: float) -> float:
        """
        코사인 거리 dist ∈ [0,2] → relevance ∈ [0,1]
        sim = 1 - dist, relevance = (sim+1)/2 = 1 - dist/2
        
        음수 거리(내적 점수)의 경우: tanh 정규화
        """
        if distance < 0:
            # 내적 점수일 경우 (높을수록 유사)
            import math
            return (math.tanh(distance) + 1) / 2
        return max(0.0, min(1.0, 1.0 - distance / 2.0))

    def _filter_and_cut(self, pairs: List[Tuple[Document, float]]) -> List[Document]:
        """
        (doc, raw_score) 목록을 relevance로 변환하고 threshold/k 적용.
        """
        ranked = []
        for doc, raw in pairs:
            # 더미 제거
            content = getattr(doc, "page_content", "")
            if "__DUMMY__INITIAL__ENTRY__" in content:
                continue
            if doc.metadata.get("is_dummy") is True:
                continue
            
            rel = self._cosine_distance_to_relevance(raw)
            print(f"📄 문서: score={rel:.3f} (raw={raw:.3f})")
            
            if (self.score_threshold is None) or (rel >= self.score_threshold):
                ranked.append((doc, rel))

        # relevance 내림차순 상위 k개
        ranked.sort(key=lambda x: x[1], reverse=True)
        filtered = [d for d, _ in ranked[: self.k]]
        print(f"✅ 최종 선택: {len(filtered)}개 문서")
        return filtered

    # 🔥 동기 메서드
    def invoke(self, query: str) -> List[Document]:
        """LangChain 표준 동기 메서드"""
        try:
            pairs = self.vector_store.similarity_search_with_score(query, k=self.prefetch)
            return self._filter_and_cut(pairs)
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            # 폴백: base retriever 사용
            try:
                docs = self.base_retriever.invoke(query) if hasattr(self.base_retriever, 'invoke') else []
                cleaned = [d for d in docs if "__DUMMY__INITIAL__ENTRY__" not in getattr(d, "page_content", "")]
                return cleaned[: self.k]
            except:
                return []

    # 🔥 비동기 메서드 추가
    async def ainvoke(self, query: str) -> List[Document]:
        """LangChain 표준 비동기 메서드"""
        try:
            # similarity_search_with_score는 동기 함수 → 스레드풀에서 실행
            loop = asyncio.get_event_loop()
            pairs = await loop.run_in_executor(
                None,
                self.vector_store.similarity_search_with_score,
                query,
                self.prefetch
            )
            return self._filter_and_cut(pairs)
        except Exception as e:
            print(f"❌ 비동기 검색 오류: {e}")
            # 폴백: base retriever의 비동기 호출
            try:
                if hasattr(self.base_retriever, 'ainvoke'):
                    docs = await self.base_retriever.ainvoke(query)
                else:
                    # 동기 함수를 비동기로 실행
                    loop = asyncio.get_event_loop()
                    docs = await loop.run_in_executor(None, self.base_retriever.invoke, query)
                
                cleaned = [d for d in docs if "__DUMMY__INITIAL__ENTRY__" not in getattr(d, "page_content", "")]
                return cleaned[: self.k]
            except Exception as e2:
                print(f"❌ 폴백도 실패: {e2}")
                return []

    # 하위 호환성 메서드 (선택사항)
    def get_relevant_documents(self, query: str) -> List[Document]:
        """하위 호환성을 위한 별칭"""
        return self.invoke(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """하위 호환성을 위한 별칭"""
        return await self.ainvoke(query)


def get_retriever_with_threshold(vector_store, k: int = 4, score_threshold: float = 0.7):
    """
    권장: Top-k 기반 베이스 리트리버 + 임계치/정규화는 래퍼에서 처리
    """
    if vector_store is None:
        raise ValueError("Vector store must be initialized before creating a retriever")

    base = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k * 2},  # prefetch를 위해 더 많이
    )
    print(f"🔍 Retriever 초기화 (k={k}, threshold={score_threshold})")
    return ThresholdWrapperRetriever(
        base, 
        vector_store, 
        k=k, 
        score_threshold=score_threshold, 
        prefetch_factor=2
    )