from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent (Agent A).

    Characteristics:
    - Within-session / within-thread memory only.
    - No persistent User.md storage.
    - Naive across sessions: completely forgets long-term facts in new threads.
    - No compact memory: accumulates all previous messages in prompt context.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                last_msg = result["messages"][-1]
                response_text = getattr(last_msg, "content", str(last_msg))

                session = self.sessions.setdefault(thread_id, SessionState())
                turn_prompt_tokens = (
                    sum(estimate_tokens(m.get("content", "")) for m in session.messages)
                    + estimate_tokens(message)
                )
                session.prompt_tokens_processed += turn_prompt_tokens
                turn_tokens = estimate_tokens(response_text)
                session.token_usage += turn_tokens

                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": response_text})

                return {
                    "response": response_text,
                    "content": response_text,
                    "token_usage": session.token_usage,
                    "prompt_tokens_processed": session.prompt_tokens_processed,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent response token count for one thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt context tokens processed for one thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline agent has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply logic for BaselineAgent."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Cumulative prompt tokens: all prior session messages + current message
        turn_prompt_tokens = (
            sum(estimate_tokens(m["content"]) for m in session.messages)
            + estimate_tokens(message)
        )
        session.prompt_tokens_processed += turn_prompt_tokens

        # Record incoming user turn
        prior_messages = list(session.messages)
        session.messages.append({"role": "user", "content": message})

        # Generate response based strictly on within-thread history
        lower_msg = message.lower()
        is_question = (
            "?" in message
            or any(
                kw in lower_msg
                for kw in [
                    "nhắc lại",
                    "tên gì",
                    "ở đâu",
                    "nghề gì",
                    "đồ uống",
                    "món ăn",
                    "style",
                    "nuôi con gì",
                    "bạn biết",
                    "tóm tắt ngắn",
                ]
            )
        )

        if is_question:
            # Baseline only knows facts mentioned earlier in THIS specific thread
            thread_text = " ".join(m["content"] for m in prior_messages)
            known_in_thread: list[str] = []

            # Check if name was mentioned in this thread
            m_name = re.search(
                r"(?:tên là|tên mình là|tên)\s+([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+)?)(?:[,.]|\s+và|\s+hiện|\s+đang|\s+ở|$)",
                thread_text,
                re.IGNORECASE,
            )
            if m_name:
                extracted_name = m_name.group(1).strip()
                if extracted_name.lower() not in ["gì", "ai", "bạn"]:
                    known_in_thread.append(f"tên bạn là {extracted_name}")

            if "cà phê sữa đá" in thread_text.lower():
                known_in_thread.append("đồ uống yêu thích là cà phê sữa đá")
            if "mì quảng" in thread_text.lower():
                known_in_thread.append("món ăn yêu thích là mì Quảng")
            if "huế" in thread_text.lower():
                known_in_thread.append("nơi ở là Huế")
            elif "đà nẵng" in thread_text.lower():
                known_in_thread.append("nơi ở là Đà Nẵng")
            if "mlops engineer" in thread_text.lower():
                known_in_thread.append("nghề nghiệp là MLOps engineer")
            elif "backend engineer" in thread_text.lower():
                known_in_thread.append("nghề nghiệp là backend engineer")

            if known_in_thread:
                response_text = (
                    "Trong phiên này, mình nhớ bạn đã chia sẻ: "
                    + ", ".join(known_in_thread)
                    + "."
                )
            else:
                response_text = (
                    "Xin lỗi, trong phiên hội thoại này mình chưa có thông tin nào trước đó về bạn. "
                    "Bạn có thể chia sẻ lại tên, nghề nghiệp hoặc sở thích của mình không?"
                )
        else:
            response_text = "Đã nhận thông tin của bạn trong phiên làm việc này. Mình sẽ lưu ý trong thread này."

        turn_tokens = estimate_tokens(response_text)
        session.token_usage += turn_tokens
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "content": response_text,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire a chat model with LangGraph MemorySaver for within-session memory."""
        try:
            if not self.config.model.api_key:
                return None
            chat_model = build_chat_model(self.config.model)
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            checkpointer = MemorySaver()
            agent = create_react_agent(chat_model, tools=[], checkpointer=checkpointer)
            return agent
        except Exception:
            return None
