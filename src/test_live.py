from __future__ import annotations

import sys
from pathlib import Path

# Configure utf-8 stdout to avoid Windows console encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_advanced import AdvancedAgent
from config import load_config


def main() -> None:
    config = load_config()
    print("==================================================")
    print("      KIỂM TRA KẾT NỐI OPENROUTER (LIVE LLM)      ")
    print("==================================================")
    print(f"Provider: {config.model.provider}")
    print(f"Model:    {config.model.model_name}")
    api_key = config.model.api_key or ""
    masked_key = (
        (api_key[:8] + "..." + api_key[-4:])
        if len(api_key) > 12
        else ("(chưa nhập)" if not api_key else "***")
    )
    print(f"API Key:  {masked_key}")
    print("==================================================\n")

    if not api_key or "your_openrouter_api_key_here" in api_key:
        print("[!] BẠN CHƯA CẬP NHẬT OPENROUTER_API_KEY TRONG FILE .env!")
        print("    Vui lòng mở file .env và điền API key thật vào OPENROUTER_API_KEY.")
        return

    print("Đang gửi tin nhắn thử nghiệm đến OpenRouter...")
    try:
        agent = AdvancedAgent(config=config, force_offline=False)

        # 1. Turn 1: Introduce information
        res = agent.reply(
            user_id="user_live_test",
            thread_id="thread_test_1",
            message="Xin chào! Mình tên là DũngCT, hiện đang ở Đà Nẵng. Hãy chào lại mình ngắn gọn nhé!",
        )
        print("\n[+] PHẢN HỒI TỪ LLM THẬT (LƯỢT 1):")
        print(res.get("response", ""))

        print("\n[+] ĐÃ LƯU PROFILE VÀO User.md:")
        print(agent.profile_store.read_text("user_live_test"))

        # 2. Turn 2: Test cross-session recall in a brand new thread
        print("Đang thử nghiệm Cross-session Recall ở thread mới...")
        recall_res = agent.reply(
            user_id="user_live_test",
            thread_id="fresh_thread_recall",
            message="Tên mình là gì và hiện tại mình đang ở đâu?",
        )
        print("\n[+] PHẢN HỒI TỪ LLM THẬT (CROSS-SESSION RECALL):")
        print(recall_res.get("response", ""))

        print("\n=> KẾT NỐI VÀ VẬN HÀNH OPENROUTER THÀNH CÔNG!")
    except Exception as e:
        print(f"\n[X] LỖI KHI GỌI OPENROUTER: {e}")


if __name__ == "__main__":
    main()
