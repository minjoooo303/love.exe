# chain.py
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os
from prompts import SYSTEM_PROMPT, CONDENSE_QUESTION_PROMPT, QA_PROMPT
from retriever import get_retriever_with_threshold
from memory import get_memory

class ConversationChain:
    def __init__(self, vector_store=None):
        # gemini 설정
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key, transport="rest")
        
        # Set up the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        
        try:
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config,
                # safety_settings=safety_settings
            )
        except Exception as e:
            print(f"Gemini 모델 초기화 중 오류 발생: {e}")
            print("사용 가능한 모델 목록을 확인합니다...")
            try:
                models = genai.list_models()
                print("사용 가능한 모델:")
                for model in models:
                    print(f"- {model.name}")
            except Exception as e2:
                print(f"모델 목록 조회 중 오류 발생: {e2}")
            raise
        
        # self.chat = self.model.start_chat(history=[])
        self.memory = get_memory()
        if vector_store is None:
            raise ValueError("Vector store가 초기화되지 않았습니다.")
        self.retriever = get_retriever_with_threshold(vector_store)
    
    async def _get_relevant_documents(self, query: str) -> List[str]:
        """검색을 위한 독립적인 질문으로 변환하고 관련 문서를 검색합니다."""
        # 독립적인 질문으로 변환
        chat_history = self.memory.load_memory_variables().get("chat_history", [])
        chat_history_str = "\n".join(f"{m.role}: {m.content}" for m in chat_history) if chat_history else ""
        standalone_query_prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=chat_history_str,
            question=query,
        )
        try:
            response = self.model.generate_content(standalone_query_prompt)
            standalone_query = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            if not standalone_query:
                standalone_query = query
        except Exception as e:
            print(f"⚠️ 독립적 질문 변환 실패: {e}, 원본 질문 사용")
            standalone_query = query 
        
        # 독립적인 질문으로 문서 검색
        docs = await self.retriever.ainvoke(standalone_query)
        return [getattr(doc, "page_content", str(doc)) for doc in docs]

    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """대화형 체인을 실행합니다."""
        try:
            query = inputs["input"]
            chat_history = self.memory.load_memory_variables().get("chat_history", [])
            
            # 관련 문서 검색 및 더미 문서 제외
            relevant_docs = await self._get_relevant_documents(query)
            relevant_docs = [d for d in relevant_docs if "__DUMMY__INITIAL__ENTRY__" not in d]
            context = "\n".join(relevant_docs) if relevant_docs else ""
            
            # 프롬프트 구성
            chat_history_str = "\n".join([
                f"{msg.role}: {msg.content}" for msg in chat_history
            ]) if chat_history else ""
            
            qa_body = QA_PROMPT.format(
                chat_history=chat_history_str,
                context=context,
                question=query,
            )
            full_prompt = SYSTEM_PROMPT+ "\n\n" + qa_body

            # Gemini로 응답 생성
            try:
                response = self.model.generate_content(full_prompt)
                if hasattr(response, 'text'):
                    ai_message = response.text
                else:
                    ai_message = str(response)
            except Exception as e:
                print(f"Error generating content: {str(e)}")
                ai_message = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
            
            # 메모리에 대화 저장
            self.memory.save_context(
                {"input": query},
                {"output": ai_message}
            )
            
            return {
                "output": ai_message,
                "source_documents": relevant_docs
            }
            
        except Exception as e:
            print(f"Error in conversation chain: {str(e)}")
            return {"output": "대화 처리 중에 오류가 발생했습니다. 다시 시도해 주세요."}

def get_conversational_chain(vector_store=None):
    """대화형 체인을 초기화하고 반환합니다."""
    chain = ConversationChain(vector_store=vector_store)
    print("대화 체인이 초기화되었습니다.")
    return chain
