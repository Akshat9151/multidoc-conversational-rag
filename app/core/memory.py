import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

from app.models.schemas import ChatMessage, Citation

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """
    In-memory session manager that preserves multi-turn dialogue history,
    supports sliding-window context injection into prompts,
    and enables follow-up query contextualization.
    """

    def __init__(self, max_history_turns: int = 6):
        self.max_history_turns = max_history_turns
        # Maps session_id -> List[ChatMessage]
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Returns existing session_id or generates a new timestamped ID."""
        if not session_id or not session_id.strip():
            session_id = f"session_{int(time.time()*1000)}"
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return session_id

    def add_user_message(self, session_id: str, message: str):
        """Records a user question in session memory."""
        session_id = self.get_or_create_session(session_id)
        chat_msg = ChatMessage(
            role="user",
            content=message.strip(),
            timestamp=datetime.utcnow().isoformat(),
        )
        self._sessions[session_id].append(chat_msg)

    def add_assistant_message(
        self, session_id: str, message: str, citations: Optional[List[Citation]] = None
    ):
        """Records an assistant response with attached citations in session memory."""
        session_id = self.get_or_create_session(session_id)
        chat_msg = ChatMessage(
            role="assistant",
            content=message.strip(),
            timestamp=datetime.utcnow().isoformat(),
            citations=citations or [],
        )
        self._sessions[session_id].append(chat_msg)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Returns the full conversation history for a given session."""
        return self._sessions.get(session_id, [])

    def get_context_window(self, session_id: str, max_turns: Optional[int] = None) -> List[ChatMessage]:
        """Returns the recent N messages within the sliding window."""
        history = self.get_history(session_id)
        limit = max_turns or self.max_history_turns
        # Each turn is (user + assistant) pair, so limit * 2 messages
        return history[-(limit * 2):]

    def format_history_for_prompt(self, session_id: str, max_turns: Optional[int] = None) -> str:
        """
        Formats recent conversational turns into clean plaintext
        suitable for feeding into the LLM context.
        """
        window = self.get_context_window(session_id, max_turns)
        if not window:
            return ""

        formatted_lines = []
        for msg in window:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            formatted_lines.append(f"{prefix} {msg.content}")

        return "\n".join(formatted_lines)

    def clear_session(self, session_id: str) -> bool:
        """Clears conversational memory for a given session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session history for: {session_id}")
            return True
        return False

# Global memory singleton
memory_manager = ConversationMemoryManager()
