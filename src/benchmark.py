from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return fraction of expected key substrings present in the answer."""
    if not expected:
        return 1.0
    lower_ans = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in lower_ans)
    return round(matches / len(expected), 3)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight response quality score (0.0 to 1.0)."""
    if not answer or "xin lỗi" in answer.lower():
        return 0.1
    score = 0.4
    rec = recall_points(answer, expected)
    score += 0.5 * rec
    if "-" in answer or "*" in answer:
        score += 0.1
    return round(min(1.0, score), 2)


def run_agent_benchmark(
    agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: Any
) -> BenchmarkRow:
    """Evaluate one agent over conversations dataset."""
    all_threads: list[str] = []
    unique_users: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    total_convs = len(conversations)
    for c_idx, conv in enumerate(conversations):
        user_id = conv.get("user_id", "user")
        thread_id = conv.get("id", "thread_default")
        unique_users.add(user_id)
        all_threads.append(thread_id)

        turns = conv.get("turns", [])
        print(f"[{agent_name}] Đang xử lý conv {c_idx + 1}/{total_convs} ({len(turns)} turns)...", end="\r", flush=True)

        # 1. Feed conversation turns in this thread
        for turn in turns:
            agent.reply(user_id, thread_id, turn)

        # 2. Ask recall questions in fresh threads (cross-session evaluation)
        recall_questions = conv.get("recall_questions", [])
        for idx, q in enumerate(recall_questions):
            recall_thread_id = f"{thread_id}_recall_{idx}"
            all_threads.append(recall_thread_id)
            question_text = q.get("question", "")
            expected = q.get("expected_contains", [])

            resp = agent.reply(user_id, recall_thread_id, question_text)
            ans = resp.get("response", "")
            pts = recall_points(ans, expected)
            qual = heuristic_quality(ans, expected)

            recall_scores.append(pts)
            quality_scores.append(qual)

    print(f"[{agent_name}] Hoàn thành {total_convs}/{total_convs} convs.                                ")

    # Aggregate metrics
    agent_tokens = sum(agent.token_usage(t) for t in all_threads)
    prompt_tokens = sum(agent.prompt_token_usage(t) for t in all_threads)
    avg_recall = (
        round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0
    )
    avg_quality = (
        round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0
    )

    if hasattr(agent, "memory_file_size"):
        memory_bytes = sum(agent.memory_file_size(u) for u in unique_users)
    else:
        memory_bytes = 0

    compactions = sum(agent.compaction_count(t) for t in all_threads)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")

    for r in rows:
        row_cells = [
            r.agent_name,
            str(r.agent_tokens_only),
            str(r.prompt_tokens_processed),
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality:.2f}",
            str(r.memory_growth_bytes),
            str(r.compactions),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def main() -> None:
    """Run Standard Benchmark and Long-Context Stress Benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Memory Systems Benchmark")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run benchmark using live LLM provider configured in .env",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to evaluate (useful for fast live testing)",
    )
    args = parser.parse_args()
    force_offline = not args.live

    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)

    conv_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    mode_str = "LIVE LLM MODE" if not force_offline else "DETERMINISTIC OFFLINE MODE"

    print("================================================================================")
    print("      PHASE 2 - TRACK 3 - DAY 17: MEMORY SYSTEMS BENCHMARK EVALUATION          ")
    print("================================================================================")
    print(f"Mode: {mode_str} | Provider: {config.model.provider} | Model: {config.model.model_name}\n")

    # 1. Standard Benchmark
    if conv_path.exists():
        standard_convs = load_conversations(conv_path)
        if args.limit:
            standard_convs = standard_convs[: args.limit]
        print("### 1. Standard Benchmark (data/conversations.json)")
        print(f"Total conversations: {len(standard_convs)}\n")

        # Clean slate state for fair comparison
        state_baseline = config.state_dir / "state_baseline_std"
        if state_baseline.exists():
            shutil.rmtree(state_baseline, ignore_errors=True)
        state_baseline.mkdir(parents=True, exist_ok=True)
        config.state_dir = state_baseline
        baseline_agent = BaselineAgent(config=config, force_offline=force_offline)

        state_advanced = config.state_dir / "state_advanced_std"
        if state_advanced.exists():
            shutil.rmtree(state_advanced, ignore_errors=True)
        state_advanced.mkdir(parents=True, exist_ok=True)
        config.state_dir = state_advanced
        advanced_agent = AdvancedAgent(config=config, force_offline=force_offline)

        row_base = run_agent_benchmark("Baseline Agent", baseline_agent, standard_convs, config)
        row_adv = run_agent_benchmark("Advanced Agent", advanced_agent, standard_convs, config)

        print(format_rows([row_base, row_adv]))
        print("\n")

    # 2. Long-Context Stress Benchmark
    if stress_path.exists():
        stress_convs = load_conversations(stress_path)
        print("### 2. Long-Context Stress Benchmark (data/advanced_long_context.json)")
        print(f"Total stress conversations: {len(stress_convs)}\n")

        state_baseline_stress = config.state_dir / "state_baseline_stress"
        if state_baseline_stress.exists():
            shutil.rmtree(state_baseline_stress, ignore_errors=True)
        state_baseline_stress.mkdir(parents=True, exist_ok=True)
        config.state_dir = state_baseline_stress
        baseline_stress = BaselineAgent(config=config, force_offline=force_offline)

        state_advanced_stress = config.state_dir / "state_advanced_stress"
        if state_advanced_stress.exists():
            shutil.rmtree(state_advanced_stress, ignore_errors=True)
        state_advanced_stress.mkdir(parents=True, exist_ok=True)
        config.state_dir = state_advanced_stress
        advanced_stress = AdvancedAgent(config=config, force_offline=force_offline)

        row_base_stress = run_agent_benchmark(
            "Baseline Agent", baseline_stress, stress_convs, config
        )
        row_adv_stress = run_agent_benchmark(
            "Advanced Agent", advanced_stress, stress_convs, config
        )

        print(format_rows([row_base_stress, row_adv_stress]))
        print("\n")

    print("Benchmark evaluation completed.")


if __name__ == "__main__":
    main()
