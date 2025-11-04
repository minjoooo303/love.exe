# web_app.py
import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from vector_store import initialize_vector_store, add_story_to_vector_store
from chain import get_conversational_chain

# ===== 환경 변수 로드 =====
load_dotenv()

# ===== FastAPI 앱 =====
app = FastAPI(title="연애 상담 챗봇")

# ===== 전역 인스턴스 =====
vector_store = None
conversation_chain = None

# ===== 경로/로그 =====
LOG_DIR = "logs"
CHAT_LOG_FILE = os.path.join(LOG_DIR, "chat_log.json")
os.makedirs(LOG_DIR, exist_ok=True)


# ===== 요청 모델 =====
class ChatRequest(BaseModel):
    message: str


class StoryRequest(BaseModel):
    content: str


# ===== 유틸: 체인/벡터스토어 지연 초기화 =====
def ensure_initialized():
    """vector_store / conversation_chain을 최초 사용 시 초기화"""
    global vector_store, conversation_chain
    if vector_store is None:
        vector_store = initialize_vector_store()
    if conversation_chain is None:
        conversation_chain = get_conversational_chain(vector_store)


def serialize_sources(source_documents):
    """
    LangChain Document 등을 문자열로 안전 변환.
    프론트/로그에 JSON 직렬화 가능한 형식으로 반환.
    """
    safe = []
    for d in (source_documents or []):
        try:
            txt = getattr(d, "page_content", None)
            if txt is None:
                # 딕셔너리 가능성
                if isinstance(d, dict) and "page_content" in d:
                    txt = d["page_content"]
                else:
                    txt = str(d)
            safe.append(str(txt))
        except Exception:
            safe.append(str(d))
    return safe


def log_interaction(user_input: str, ai_response: str, retrieved_sources: list = None):
    """대화/응답/출처 로그 저장(JSON)"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "ai_response": ai_response,
        "retrieved_sources": retrieved_sources or [],
    }

    # 파일 없으면 빈 배열로 초기화
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)

    # 안전하게 읽고 덮어쓰기
    with open(CHAT_LOG_FILE, "r+", encoding="utf-8") as f:
        try:
            content = f.read()
            logs = json.loads(content) if content else []
        except json.JSONDecodeError:
            logs = []
        logs.append(entry)
        f.seek(0)
        json.dump(logs, f, ensure_ascii=False, indent=2)
        f.truncate()


# ===== 서버 시작 메시지 =====
@app.on_event("startup")
async def startup_event():
    print("🚀 서버 시작 완료! 벡터 스토어와 체인은 첫 사용 시 자동으로 로드됩니다.")


# ===== HTML =====
@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_content = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>💕 연애 상담 챗봇</title>
<style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Roboto','Helvetica Neue',Arial,sans-serif;
        background:#f5f7fa;min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px
    }
    .container{
        background:#fff;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.08);
        max-width:900px;width:100%;height:90vh;display:flex;flex-direction:column;overflow:hidden
    }
    .header{
        background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
        color:#fff;padding:24px 32px;text-align:center;position:relative
    }
    .header::after{
        content:'';position:absolute;bottom:0;left:0;right:0;height:4px;
        background:linear-gradient(90deg,#ec4899,#f43f5e,#ec4899);
        background-size:200% 100%;animation:gradient 3s ease infinite
    }
    @keyframes gradient{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
    .header h1{font-size:26px;font-weight:700;margin-bottom:6px;letter-spacing:-.5px}
    .header p{font-size:14px;opacity:.95;font-weight:400}
    .chat-container{
        flex:1;overflow-y:auto;padding:24px;background:#fafbfc;scroll-behavior:smooth
    }
    .chat-container::-webkit-scrollbar{width:8px}
    .chat-container::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:4px}
    .chat-container::-webkit-scrollbar-thumb:hover{background:#9ca3af}
    .message{margin-bottom:20px;display:flex;animation:slideIn .4s cubic-bezier(.16,1,.3,1)}
    @keyframes slideIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
    .message.user{justify-content:flex-end}
    .message-content{
        max-width:75%;padding:14px 18px;border-radius:16px;word-wrap:break-word;line-height:1.6;font-size:15px
    }
    .message.user .message-content{
        background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
        color:#fff;border-bottom-right-radius:4px;box-shadow:0 2px 8px rgba(99,102,241,.3)
    }
    .message.bot .message-content{
        background:#fff;color:#1f2937;border:1px solid #e5e7eb;border-bottom-left-radius:4px;
        box-shadow:0 1px 3px rgba(0,0,0,.05)
    }
    .sources{margin-top:10px;padding:10px;background:#f9fafb;border-left:3px solid #6366f1;border-radius:5px;font-size:12px;color:#6b7280}
    .sources-title{font-weight:600;margin-bottom:5px;color:#6366f1}
    .input-container{padding:20px 24px 24px;background:#fff;border-top:1px solid #e5e7eb}
    .input-wrapper{display:flex;gap:12px;margin-bottom:12px}
    input[type="text"]{
        flex:1;padding:14px 20px;border:2px solid #e5e7eb;border-radius:12px;font-size:15px;outline:none;transition:.2s;font-family:inherit
    }
    input[type="text"]:focus{border-color:#6366f1;box-shadow:0 0 0 3px rgba(99,102,241,.1)}
    input[type="text"]::placeholder{color:#9ca3af}
    button{
        padding:14px 28px;border:none;border-radius:12px;font-size:15px;font-weight:600;cursor:pointer;transition:.2s;white-space:nowrap;font-family:inherit
    }
    .send-btn{background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);color:#fff}
    .send-btn:hover{transform:translateY(-1px);box-shadow:0 8px 16px rgba(99,102,241,.3)}
    .send-btn:active{transform:translateY(0)}
    .add-story-btn,.clear-btn{
        background:#f3f4f6;color:#6b7280;font-size:13px;padding:10px 18px;border:1px solid #e5e7eb
    }
    .add-story-btn:hover{background:#e5e7eb;color:#374151}
    .clear-btn:hover{background:#fee2e2;color:#dc2626;border-color:#fecaca}
    .button-group{display:flex;gap:8px;justify-content:center}
    .typing-indicator{display:flex;gap:5px;padding:10px}
    .typing-indicator span{width:8px;height:8px;border-radius:50%;background:#9ca3af;animation:typing 1.4s infinite}
    .typing-indicator span:nth-child(2){animation-delay:.2s}
    .typing-indicator span:nth-child(3){animation-delay:.4s}
    @keyframes typing{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-10px)}}
    @media (max-width:768px){
        .container{height:100vh;border-radius:0}
        .message-content{max-width:85%}
        .input-wrapper{flex-direction:column}
        button{width:100%}
    }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>연애 상담 챗봇</h1>
        <p>따뜻하고 진심어린 조언을 드립니다</p>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="message bot">
            <div class="message-content">
                안녕하세요! 연애 상담 챗봇입니다.<br>
                어떤 고민이 있으신가요? 편하게 말씀해주세요.
            </div>
        </div>
    </div>

    <div class="input-container">
        <div class="input-wrapper">
            <input
                type="text"
                id="messageInput"
                placeholder="메시지를 입력하세요..."
                onkeypress="handleKeyPress(event)"
            >
            <button class="send-btn" onclick="sendMessage()">전송 📤</button>
        </div>
        <div class="button-group">
            <!-- event를 전달해야 함 -->
            <button class="add-story-btn" onclick="toggleStoryMode(event)">사연 추가 ➕</button>
            <button class="clear-btn" onclick="clearMemory()">대화 초기화 🔄</button>
        </div>
    </div>
</div>

<script>
let isStoryMode = false;

function handleKeyPress(event){
    if(event.key === 'Enter'){ sendMessage(); }
}

function setStoryModeUI(on, btn){
    const input = document.getElementById('messageInput');
    isStoryMode = on;
    if(on){
        input.placeholder = '사연을 입력하세요... (다시 클릭하면 일반 모드)';
        btn.textContent = '사연 모드 ON ✅';
        btn.style.background = '#fef3c7';
        btn.style.color = '#92400e';
        btn.style.borderColor = '#fde68a';
    }else{
        input.placeholder = '메시지를 입력하세요...';
        btn.textContent = '사연 추가 ➕';
        btn.style.background = '#f3f4f6';
        btn.style.color = '#6b7280';
        btn.style.borderColor = '#e5e7eb';
    }
}

function toggleStoryMode(e){
    const btn = e?.target || document.querySelector('.add-story-btn');
    setStoryModeUI(!isStoryMode, btn);
}

function showTypingIndicator(){
    const chatContainer = document.getElementById('chatContainer');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typing-' + Date.now();

    const indicator = document.createElement('div');
    indicator.className = 'message-content';
    indicator.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    typingDiv.appendChild(indicator);
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return typingDiv.id;
}

function removeTypingIndicator(id){
    const el = document.getElementById(id);
    if(el) el.remove();
}

function addMessage(text, sender, sources=null){
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = (text || '').replace(/\n/g,'<br>');

    if(sources && sources.length > 0){
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = '<div class="sources-title">📚 참고한 사연:</div>';
        sources.forEach((source, idx) => {
            const s = String(source);
            const preview = s.substring(0, 100) + (s.length > 100 ? '...' : '');
            sourcesDiv.innerHTML += `<div>${idx + 1}. ${preview}</div>`;
        });
        content.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage(){
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if(!message) return;

    addMessage(message, 'user');
    input.value = '';

    const typingId = showTypingIndicator();

    try{
        const url = isStoryMode ? '/add-story' : '/chat';
        const payload = isStoryMode ? {content: message} : {message};
        const res = await fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        removeTypingIndicator(typingId);

        if(isStoryMode){
            addMessage(data.message, 'bot');
            // 사연 전송 후에는 항상 OFF로 확정
            setStoryModeUI(false, document.querySelector('.add-story-btn'));
        }else{
            addMessage(data.response, 'bot', data.sources);
        }
    }catch(err){
        removeTypingIndicator(typingId);
        console.error(err);
        addMessage('죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.', 'bot');
    }
}

async function clearMemory(){
    if(!confirm('대화 기록을 초기화하시겠습니까?')) return;
    try{
        const res = await fetch('/clear', {method:'POST'});
        await res.json();
        document.getElementById('chatContainer').innerHTML = `
            <div class="message bot">
                <div class="message-content">
                    대화 기록이 초기화되었습니다. 새로운 대화를 시작해주세요! 😊
                </div>
            </div>
        `;
    }catch(err){
        console.error(err);
        alert('초기화 중 오류가 발생했습니다.');
    }
}
</script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ===== API =====
@app.post("/chat")
async def chat(request: ChatRequest):
    """채팅 메시지 처리"""
    try:
        ensure_initialized()
        response = await conversation_chain.ainvoke({"input": request.message})

        # 체인 구현에 따라 키가 다를 수 있어 대비
        ai_message = response.get("output", "") or response.get("answer", "") or ""
        source_documents = response.get("source_documents", [])
        sources_text = serialize_sources(source_documents)

        # 로그 (문자열만)
        log_interaction(request.message, ai_message, sources_text)

        return JSONResponse({"response": ai_message, "sources": sources_text})
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-story")
async def add_story(request: StoryRequest):
    """사연 추가"""
    try:
        ensure_initialized()
        story_id = str(uuid.uuid4())
        add_story_to_vector_store(vector_store, request.content, story_id, persist=True)
        
        return JSONResponse({
            "message": f"사연이 성공적으로 추가되었습니다! 🎉\n(ID: {story_id[:8]}...)",
            "story_id": story_id
        })
    except Exception as e:
        print(f"Add story error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_memory():
    """메모리 초기화"""
    try:
        ensure_initialized()
        conversation_chain.memory.clear()
        return JSONResponse({"message": "메모리가 초기화되었습니다."})
    except Exception as e:
        print(f"Clear memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)