
# main.py
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

from vector_store import initialize_vector_store, add_story_to_vector_store
from chain import get_conversational_chain

# Load environment variables
load_dotenv()

# Global instances for vector store and conversational chain
# These will be initialized once when the application starts
vector_store = None
conversation_chain = None

# Path for chat logs
LOG_DIR = "logs"
CHAT_LOG_FILE = os.path.join(LOG_DIR, "chat_log.json")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def log_interaction(user_input: str, ai_response: str, retrieved_sources: list = None):
    """
    Logs user interactions, AI responses, and retrieved sources to a JSON file.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "ai_response": ai_response,
        "retrieved_sources": retrieved_sources if retrieved_sources else []
    }
    # Append to JSON file
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f) # Initialize with empty list

    with open(CHAT_LOG_FILE, "r+", encoding="utf-8") as f:
        file_content = f.read()
        if file_content:
            logs = json.loads(file_content)
        else:
            logs = []
        logs.append(log_entry)
        f.seek(0) # Rewind to beginning
        json.dump(logs, f, ensure_ascii=False, indent=4)
        f.truncate() # Truncate any remaining old content
    print("대화가 로그에 저장되었습니다.")

async def initialize_application():
    global vector_store, conversation_chain
    print("애플리케이션 시작: 벡터 스토어 및 대화 체인 초기화 중...")
    vector_store = initialize_vector_store()
    conversation_chain = get_conversational_chain(vector_store)
    print("초기화 완료.")

async def add_story_cli(story_content: str):
    if not vector_store:
        await initialize_application() # Ensure vector store is initialized

    story_id = str(uuid.uuid4()) # Generate a unique ID for the story
    add_story_to_vector_store(vector_store, story_content, story_id, persist=True)
    print(f"사연이 성공적으로 추가되었습니다. (ID: {story_id})")

async def chat_cli(question: str):
    if not conversation_chain:
        await initialize_application() # Ensure conversation chain is initialized

    # print(f"사용자 질문: {question}")
    try:
        response = await conversation_chain.ainvoke({"input": question})
        
        ai_message = response.get("output", "")
        source_documents = response.get("source_documents", [])

        print(f"AI 응답: {ai_message}")
        if source_documents:
            print("참고 사연:")
            for doc in source_documents:
                print(f"- {doc}")

        log_interaction(question, ai_message, source_documents)
        return {"answer": ai_message, "source_documents": source_documents}
    except Exception as e:
        print(f"대화 중 오류 발생: {e}")
        return {"answer": f"오류 발생: {e}", "source_documents": []}

async def main():
    await initialize_application()
    print("--------------------------------------------------")
    print("CLI 모드 시작. 'exit' 또는 'quit'를 입력하여 종료하세요.")
    print("사연 추가: 'add_story: 당신의 사연 내용' 형식으로 입력하세요.")
    print("대화 기록 초기화: 'clear' 또는 'reset'을 입력하세요.")
    print("--------------------------------------------------")

    while True:
        user_input = input("당신의 질문: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("CLI 모드를 종료합니다.")
            break
        elif user_input.lower() in ["clear", "reset"]:
            if conversation_chain:
                conversation_chain.memory.clear()
                print("대화 기록이 초기화되었습니다.")
            else:
                print("대화 체인이 초기화되지 않았습니다.")
        elif user_input.lower().startswith("add_story:"):
            story_content = user_input[len("add_story:"):].strip()
            if story_content:
                await add_story_cli(story_content)
            else:
                print("사연 내용을 입력해주세요. 예: add_story: 내 사연 내용")
        elif user_input:
            await chat_cli(user_input)
        else:
            print("입력된 내용이 없습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


