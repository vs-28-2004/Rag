"""Chat history management: storage, formatting for prompts, and export."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))
    citations: list[str] = field(default_factory=list)


class ChatMemory:
    """Holds the running conversation and provides formatting utilities."""

    def __init__(self, max_turns_in_context: int = 6):
        self.messages: list[ChatMessage] = []
        self.max_turns_in_context = max_turns_in_context

    def add(self, role: str, content: str, citations: list[str] | None = None) -> None:
        self.messages.append(ChatMessage(role=role, content=content, citations=citations or []))

    def clear(self) -> None:
        self.messages = []

    def as_prompt_history(self) -> str:
        """Return the last N turns formatted for inclusion in the LLM prompt."""
        recent = self.messages[-(self.max_turns_in_context * 2):]
        lines = [f"{m.role.capitalize()}: {m.content}" for m in recent]
        return "\n".join(lines)

    def export_txt(self) -> str:
        lines = []
        for m in self.messages:
            lines.append(f"[{m.timestamp}] {m.role.upper()}: {m.content}")
            if m.citations:
                lines.append("  Sources: " + "; ".join(m.citations))
        return "\n\n".join(lines)

    def export_markdown(self) -> str:
        lines = ["# RAGenius AI — Chat Export\n"]
        for m in self.messages:
            speaker = "🧑 User" if m.role == "user" else "🤖 RAGenius"
            lines.append(f"### {speaker} · `{m.timestamp}`\n\n{m.content}\n")
            if m.citations:
                lines.append("**Sources:**\n" + "\n".join(f"- {c}" for c in m.citations) + "\n")
        return "\n".join(lines)


@dataclass
class ChatSession:
    """A single named conversation, wrapping its own ChatMemory."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = "New chat"
    memory: ChatMemory = field(default_factory=ChatMemory)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def maybe_set_title(self, first_user_message: str, max_len: int = 42) -> None:
        """Derive a short title from the first user message, once."""
        if self.title != "New chat":
            return
        text = first_user_message.strip().replace("\n", " ")
        self.title = (text[:max_len] + "…") if len(text) > max_len else (text or "New chat")


class ChatSessionStore:
    """Holds multiple ChatSessions ('recent conversations') and tracks the active one."""

    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.active_id: str | None = None
        self.new_session()

    def new_session(self) -> ChatSession:
        session = ChatSession()
        self.sessions[session.id] = session
        self.active_id = session.id
        return session

    @property
    def active(self) -> ChatSession:
        return self.sessions[self.active_id]

    def switch_to(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.active_id = session_id

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if not self.sessions or self.active_id == session_id:
            self.new_session()

    def ordered_sessions(self) -> list[ChatSession]:
        """Most-recently-created first."""
        return sorted(self.sessions.values(), key=lambda s: s.created_at, reverse=True)

    def search(self, query: str) -> list[tuple[ChatSession, ChatMessage]]:
        """Search every session's messages for a substring match (case-insensitive)."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        hits: list[tuple[ChatSession, ChatMessage]] = []
        for session in self.sessions.values():
            for msg in session.memory.messages:
                if query_lower in msg.content.lower():
                    hits.append((session, msg))
        return hits
