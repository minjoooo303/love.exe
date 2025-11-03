연애 상담 챗봇 – RAG(Retrieval-Augmented Generation) 기반으로 과거 사연을 학습하여 맥락에 맞는 조언을 제공합니다.

💡 실사용은 개발·학습 목적이며, 전문 상담을 대체하지 않습니다.

✨ 주요 기능

💬 실시간 채팅: 연애 고민 상담

📚 사연 학습: 새로운 사연 추가 및 벡터 DB 저장

🧠 대화 기억: 이전 대화 맥락 유지

🔍 유사 사연 검색: 관련 사연 기반 답변 생성

🚀 빠른 시작
1) 저장소 클론
git clone <repository-url>
cd loveexe

2) 가상환경 생성 및 패키지 설치
conda create -n lovebot python=3.10 -y
conda activate lovebot
pip install -r requirements.txt

3) API 키 설정

루트에 .env 파일을 만들고 다음을 입력하세요.

GEMINI_API_KEY=your_gemini_api_key_here

4) 실행
python web_app.py


브라우저에서 👉 http://localhost:8000
 접속

📦 필수 패키지

fastapi

uvicorn

python-dotenv

langchain

langchain-community

langchain-huggingface

google-generativeai

faiss-cpu

sentence-transformers

임베딩 모델: sentence-transformers/all-MiniLM-L6-v2
LLM: Google Gemini 2.0 Flash

🏗️ 프로젝트 구조
loveexe/
├── web_app.py          # FastAPI 웹 서버
├── chain.py            # RAG 체인 로직
├── vector_store.py     # FAISS 벡터 저장소
├── retriever.py        # 문서 검색 및 필터링
├── memory.py           # 대화 메모리 관리
├── prompts.py          # 프롬프트 템플릿
├── .env                # API 키 설정 (로컬 전용)
├── data/               # 벡터 DB 저장소
└── logs/               # 대화 로그

🎯 사용 방법

일반 채팅: 입력창에 메시지를 쓰고 전송

사연 추가: 사연 추가 ➕ 버튼 → 사연 입력 → 저장

대화 초기화: 대화 초기화 🔄 버튼

🔧 기술 스택

Backend: FastAPI

LLM: Google Gemini 2.0 Flash

Vector DB: FAISS

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Frontend: HTML / CSS / JavaScript(인라인)
