from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with a low compact threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    model_config = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    judge_config = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=model_config,
        judge_model=judge_config,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dungct_test"

    # 1. Initial write
    initial_content = "# User Profile\n- **name**: DũngCT\n- **location**: Đà Nẵng\n"
    store.write_text(user_id, initial_content)

    # 2. Read and verify content & size
    read_back = store.read_text(user_id)
    assert "DũngCT" in read_back
    assert "Đà Nẵng" in read_back
    assert store.file_size(user_id) > 0

    # 3. Edit text (update location from Đà Nẵng to Huế)
    edited = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert edited is True

    updated_content = store.read_text(user_id)
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content

    # 4. Structured facts
    facts = store.facts(user_id)
    assert facts.get("name") == "DũngCT"
    assert facts.get("location") == "Huế"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction in AdvancedAgent."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)
    thread_id = "thread_compact_test"

    # Send multiple turns with enough tokens to exceed compact_threshold_tokens=50
    for i in range(8):
        msg = f"Lượt {i}: Đây là đoạn tin tức kỹ thuật rất dài để ép tổng số token trong hội thoại vượt ngưỡng threshold {i}."
        agent.reply("dungct_test", thread_id, msg)

    # Compaction must have been triggered at least once
    assert agent.compaction_count(thread_id) > 0

    # Compact context should have a summary and bounded recent messages
    ctx = agent.compact_memory.context(thread_id)
    assert bool(ctx.get("summary"))
    messages: list = ctx.get("messages", [])  # type: ignore
    assert len(messages) <= cfg.compact_keep_messages + 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify AdvancedAgent remembers across sessions while BaselineAgent does not."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)
    user_id = "dungct_recall_test"

    # Session 1: User introduces facts in thread_1
    intro_msg = "Chào bạn, mình tên là DũngCT và thích cà phê sữa đá."
    baseline.reply(user_id, "thread_1", intro_msg)
    advanced.reply(user_id, "thread_1", intro_msg)

    # Session 2: Fresh thread_2 asking recall questions
    recall_msg = "Mình tên gì và thích uống gì?"
    base_res = baseline.reply(user_id, "thread_2", recall_msg)
    adv_res = advanced.reply(user_id, "thread_2", recall_msg)

    # Baseline has no User.md, so it completely forgets in a new thread
    assert "DũngCT" not in base_res["response"]

    # Advanced uses persistent User.md, so it recalls accurately
    assert "DũngCT" in adv_res["response"]
    assert "cà phê sữa đá" in adv_res["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)
    thread_id = "thread_stress_load"

    # Send 10 substantial conversation turns
    for i in range(10):
        turn_text = (
            f"Lượt {i}: Báo cáo chi tiết về tiến độ dự án AI, bao gồm các mốc kiểm thử, "
            f"tối ưu hóa pipeline MLOps và đánh giá hiệu năng mô hình số {i}."
        )
        baseline.reply("user_stress", thread_id, turn_text)
        advanced.reply("user_stress", thread_id, turn_text)

    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced must have triggered compaction
    assert advanced.compaction_count(thread_id) > 0

    # Advanced prompt load must be lower than baseline due to compaction
    assert advanced_prompt_tokens < baseline_prompt_tokens
