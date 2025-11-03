
# memory.py
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Memory:
    messages: List[Message] = field(default_factory=list)
    memory_key: str = "chat_history"
    return_messages: bool = True
    max_token_limit: int = 2000  # 최대 토큰 제한 (대략 메시지 개수로 변환)
    max_messages: int = 10  # 최근 N개 메시지만 유지

    def clear(self):
        self.messages.clear()

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        """Save the current conversation context."""
        if "input" in inputs:
            self.messages.append(Message(role="user", content=inputs["input"]))
        if "output" in outputs:
            self.messages.append(Message(role="assistant", content=outputs["output"]))
        
        # 메시지가 max_messages를 초과하면 오래된 것부터 삭제
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def load_memory_variables(self) -> Dict[str, Any]:
        """Return the stored messages (최근 메시지만)."""
        if self.return_messages:
            return {self.memory_key: self.messages[-self.max_messages:]}
        else:
            return {self.memory_key: self._get_buffer_string()}

    def _get_buffer_string(self) -> str:
        """Get the buffer string of messages (최근 메시지만)."""
        recent_messages = self.messages[-self.max_messages:]
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in recent_messages
        ])

def get_memory(max_messages: int = 10):
    """Initialize and return a new memory instance."""
    return Memory(max_messages=max_messages)


