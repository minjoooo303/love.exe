## love.exe: 연애 상담 챗봇 💌

**love.exe**는 **RAG (Retrieval-Augmented Generation)** 기술을 기반으로 구현된 연애 상담 챗봇입니다. 과거의 다양한 연애 사연을 학습하여 사용자의 고민 맥락에 가장 적합한 조언을 제공합니다.

> 💡 **주의**: 본 프로젝트는 학습 및 연구용이며, 전문적인 심리 상담을 대체하지 않습니다.

-----

## ✨ 주요 기능

| 기능 | 설명 | 기술적 구현 |
| :--- | :--- | :--- |
| 💬 **실시간 채팅** | 사용자 연애 고민에 대한 실시간 상담 제공. | `web_app.py`, `chain.py` (FastAPI) |
| 📚 **사연 학습** | 새로운 연애 사연을 **벡터 DB**에 추가하고 저장. | `vector_store.py` (FAISS, Embedding) |
| 🧠 **대화 기억** | 이전 대화 내용을 기억하여 맥락에 맞는 답변 생성. | `memory.py` (LangChain Memory) |
| 🔍 **유사 사연 검색** | 사용자의 현재 고민과 유사한 과거 사연을 검색하여 답변의 근거로 활용. | `retriever.py` (RAG Pattern) |

-----

## 🏗️ 기술 스택

| 분류 | 컴포넌트 | 설명 |
| :--- | :--- | :--- |
| **LLM** | Google Gemini 2.0 Flash | 대화 생성 및 조언 제공 |
| **임베딩** | `sentence-transformers/all-MiniLM-L6-v2` | 텍스트를 벡터로 변환 |
| **벡터 DB** | `FAISS (faiss-cpu)` | 유사 사연 검색 및 저장 |
| **프레임워크** | `FastAPI`, `Uvicorn` | 웹 서버 구축 및 서비스 |
| **RAG/Orchestration** | `LangChain` | 체인 로직, 메모리, RAG 구현 |

-----

## 🚀 빠른 시작 (Quick Start)

### 1\) 저장소 클론 및 이동

```bash
git clone https://github.com/your-name/loveexe.git
cd loveexe
```

### 2\) 가상환경 설정 및 패키지 설치

Python 3.10 환경을 권장합니다.

```bash
# 가상환경 생성 및 활성화
conda create -n lovebot python=3.10 -y
conda activate lovebot

# 필수 패키지 설치
pip install -r requirements.txt
```

### 3\) API 키 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고, **Gemini API 키**를 입력하세요.

```dotenv
# .env 파일 내용
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4\) 실행

```bash
python web_app.py
```

브라우저에서 다음 주소로 접속하면 챗봇을 사용할 수 있습니다.

▶︎ **http://localhost:8000**

-----

## 📁 프로젝트 구조

```
loveexe/
├── web_app.py          # 🌐 FastAPI 웹 서버 및 엔드포인트 정의
├── chain.py            # 🧠 RAG 체인 및 LLM 호출 로직
├── vector_store.py     # 📦 FAISS 벡터 저장소 초기화 및 관리
├── retriever.py        # 🔍 문서 검색 및 필터링 로직
├── memory.py           # 💬 대화 메모리 관리 (맥락 유지)
├── prompts.py          # 📝 시스템 프롬프트 및 템플릿 정의
├── .env                # 🔑 API 키 설정 파일 (민감 정보 보호)
├── requirements.txt    # ✅ 필수 Python 패키지 목록
├── data/               # 📂 벡터 DB (FAISS 인덱스) 저장 디렉터리
└── logs/               # 📋 대화 로그 및 디버깅 기록
```
