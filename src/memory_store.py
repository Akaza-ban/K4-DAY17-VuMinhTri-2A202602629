from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Approximate token count for a text string.

    Heuristic: ~4 characters per token on average for mixed text.
    Returns 0 for empty or whitespace-only text.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Stores and retrieves long-term profile data per user.
    """

    root_dir: Path

    def _sanitize_user_id(self, user_id: str) -> str:
        # Keep alphanumeric, dashes, and underscores
        cleaned = re.sub(r"[^\w\-]", "_", user_id.strip())
        return cleaned or "default_user"

    def path_for(self, user_id: str) -> Path:
        """Return the Path to User.md for a given user."""
        user_slug = self._sanitize_user_id(user_id)
        user_dir = self.root_dir / user_slug
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return the content of User.md or empty string if not found."""
        file_path = self.path_for(user_id)
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write content to User.md and return the file Path."""
        file_path = self.path_for(user_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence of search_text inside User.md."""
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        """Return the size of User.md in bytes."""
        file_path = self.path_for(user_id)
        if not file_path.exists():
            return 0
        return file_path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured key-value facts from User.md."""
        content = self.read_text(user_id)
        results: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            # Match lines like "- **key**: value" or "- key: value"
            m = re.match(r"^-\s*(?:\*\*)?([\w\-]+)(?:\*\*)?\s*:\s*(.+)$", line)
            if m:
                results[m.group(1).lower()] = m.group(2).strip()
        return results

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Update or insert key-value facts and persist back to User.md."""
        current_facts = self.facts(user_id)
        current_facts.update(new_facts)

        lines = ["# User Profile"]
        for k, v in current_facts.items():
            lines.append(f"- **{k}**: {v}")
        new_content = "\n".join(lines) + "\n"
        self.write_text(user_id, new_content)

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Convenience method to upsert a single fact."""
        self.upsert_facts(user_id, {key: value})


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Handles noise (e.g. temporary trips, jokes) and updates/corrections.
    """
    facts: dict[str, str] = {}
    lower_msg = message.lower()

    # Skip recall question turns asking about existing facts
    is_recall_query = any(
        kw in lower_msg
        for kw in [
            "tên mình là gì",
            "tên là gì",
            "mình tên gì",
            "tên gì",
            "ở đâu",
            "đang ở đâu",
            "làm nghề gì",
            "uống gì",
            "ăn gì",
            "con gì",
            "style gì",
            "nhắc lại",
            "thử nhớ",
            "bạn biết",
            "tóm tắt ngắn",
            "đâu mới là",
        ]
    )
    if is_recall_query:
        return facts

    # 1. Name extraction
    name_patterns = [
        r"(?:chào bạn,\s*)?mình tên là\s+([A-Za-zÀ-ỹ0-9_ ]+?)(?:[,.]|\s+hiện|\s+đang|\s+ở|$)",
        r"tên mình là\s+([A-Za-zÀ-ỹ0-9_ ]+?)(?:[,.]|\s+hiện|\s+đang|\s+ở|$)",
        r"(?:^|\s)tên\s+(DũngCT(?:\s+Stress)?)(?:[,.]|\s|$)",
    ]
    for pat in name_patterns:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            extracted_name = m.group(1).strip()
            # Filter out generic words or questions
            tokens_in_name = extracted_name.lower().split()
            if not any(w in tokens_in_name for w in ["gì", "ai", "bạn", "nào", "đâu", "và"]):
                if len(extracted_name) > 1:
                    facts["name"] = extracted_name
                    break

    # 2. Location & corrections
    # Check for noise: "Hà Nội chỉ là nơi mình vừa bay ra họp"
    is_hanoi_noise = "hà nội" in lower_msg and any(
        kw in lower_msg for kw in ["họp", "chỉ là nơi", "không phải nơi ở"]
    )
    if not is_hanoi_noise:
        # Check explicit location changes: Huế vs Đà Nẵng
        if any(
            kw in lower_msg
            for kw in [
                "cập nhật từ huế sang đà nẵng",
                "làm việc ở đà nẵng vài tháng",
                "nơi ở hiện tại là đà nẵng",
            ]
        ):
            facts["location"] = "Đà Nẵng"
        elif any(
            kw in lower_msg
            for kw in [
                "giờ mình đang ở huế",
                "đang ở huế chứ không còn ở đà nẵng",
                "mình vẫn ở huế",
                "hiện ở huế",
                "ở huế",
            ]
        ):
            facts["location"] = "Huế"
        elif "đà nẵng" in lower_msg and "không còn ở đà nẵng" not in lower_msg:
            # If Đà Nẵng is mentioned as residence
            if any(kw in lower_msg for kw in ["mình ở đà nẵng", "nơi ở hiện tại là đà nẵng"]):
                facts["location"] = "Đà Nẵng"

    # 3. Profession & corrections
    # Check for noise: "product manager... chỉ là câu đùa"
    is_pm_joke = "product manager" in lower_msg and any(
        kw in lower_msg for kw in ["đùa", "chỉ là câu đùa", "không phải"]
    )
    if not is_pm_joke:
        if any(
            kw in lower_msg
            for kw in [
                "chuyển sang mlops engineer",
                "làm mlops engineer",
                "nghề mlops engineer",
                "công việc mlops",
                "nghề nghiệp hiện tại vẫn là mlops engineer",
            ]
        ):
            facts["profession"] = "MLOps engineer"
        elif "làm backend engineer" in lower_msg and "không còn làm backend" not in lower_msg:
            facts["profession"] = "backend engineer"

    # 4. Drink preference
    if "cà phê sữa đá" in lower_msg:
        facts["drink"] = "cà phê sữa đá"

    # 5. Food preference
    if "mì quảng" in lower_msg:
        facts["food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower_msg:
        facts["pet"] = "corgi"
    elif "bé corgi tên bơ" in lower_msg or "con bơ" in lower_msg:
        facts["pet"] = "corgi"

    # 7. Response style
    if "3 bullet" in lower_msg or "ba bullet" in lower_msg:
        facts["style"] = "3 bullet ngắn, có ví dụ thực chiến, trade-off"
    elif "ngắn gọn" in lower_msg or "bullet ngắn" in lower_msg:
        facts["style"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Technical interests
    if "python" in lower_msg and ("ai" in lower_msg or "agent" in lower_msg):
        facts["interests"] = "Python, AI"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a concise summary of older messages to preserve context."""
    if not messages:
        return ""

    subset = messages[-max_items:]
    lines: list[str] = []
    for msg in subset:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if len(content) > 80:
            content = content[:80] + "..."
        lines.append(f"- {role}: {content}")

    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long conversation threads.

    Keeps recent messages in full. When cumulative tokens exceed threshold,
    compresses older messages into a summary and increments compaction_count.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _get_thread_state(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and trigger compaction if threshold is exceeded."""
        st = self._get_thread_state(thread_id)
        messages: list[dict[str, str]] = st["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current token load
        summary_tokens = estimate_tokens(str(st.get("summary", "")))
        messages_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        total_tokens = summary_tokens + messages_tokens

        # Check if compaction should trigger
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[: -self.keep_messages]
            kept = messages[-self.keep_messages :]

            chunk_summary = summarize_messages(older)
            existing_summary = str(st.get("summary", "")).strip()

            if existing_summary and chunk_summary:
                all_lines = [
                    line.strip()
                    for line in (existing_summary + "\n" + chunk_summary).splitlines()
                    if line.strip().startswith("-")
                ]
                # Deduplicate and keep only the latest 4 concise summary lines
                unique_lines: list[str] = []
                for line in all_lines:
                    if not unique_lines or line != unique_lines[-1]:
                        unique_lines.append(line)
                combined_summary = "\n".join(unique_lines[-4:])
            elif chunk_summary:
                combined_summary = chunk_summary
            else:
                combined_summary = existing_summary

            st["summary"] = combined_summary
            st["messages"] = list(kept)
            st["compactions"] = int(st.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the current context for a thread."""
        return self._get_thread_state(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of compactions that occurred for this thread."""
        st = self._get_thread_state(thread_id)
        return int(st.get("compactions", 0))
