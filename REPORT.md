# Báo cáo Đánh giá Hệ thống Memory cho AI Agent (Day 17)

**Học viên:** Vũ Minh Trí  
**Mã học viên:** 2A202602629  
**Giai đoạn:** Phase 2, Track 3, Day 17: Memory Systems for AI Agent  

---

## 1. Tổng quan Kiến trúc Hệ thống

Hệ thống được thiết kế theo mô hình phân tầng memory đa lớp, giải quyết bài toán cân bằng giữa **độ ghi nhớ dài hạn (cross-session recall)** và **chi phí xử lý ngữ cảnh (prompt tokens processed)**.

Hệ thống bao gồm hai agent để so sánh đối chứng:
- **Baseline Agent (Agent A)**: Chỉ có bộ nhớ ngắn hạn (*Within-session memory*) theo từng `thread_id`. Không có bộ nhớ bền vững (*User.md*), không có cơ chế nén ngữ cảnh. Khi sang phiên làm việc mới, agent hoàn toàn không nhớ thông tin từ các phiên trước.
- **Advanced Agent (Agent B)**: Tích hợp đầy đủ 3 tầng bộ nhớ:
  1. *Short-term memory*: Bộ đệm tin nhắn gần nhất (`keep_messages`).
  2. *Persistent memory*: Lưu trữ hồ sơ người dùng bền vững qua file `state/profiles/<user>/User.md`.
  3. *Compact memory*: Cơ chế nén ngữ cảnh tự động (`CompactMemoryManager`), tóm tắt các tin nhắn cũ khi tổng token vượt ngưỡng `compact_threshold_tokens`.

---

## 2. Kết quả Kiểm thử Tự động (`pytest src/test_agents.py -v`)

Hệ thống đã vượt qua toàn bộ **4/4 bài test cốt lõi**:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
collected 4 items

src/test_agents.py::test_user_markdown_read_write_edit PASSED            [ 25%]
src/test_agents.py::test_compact_trigger PASSED                          [ 50%]
src/test_agents.py::test_cross_session_recall PASSED                     [ 75%]
src/test_agents.py::test_compact_reduces_prompt_load_on_long_thread PASSED [100%]

============================== 4 passed in 0.05s ==============================
```

- `test_user_markdown_read_write_edit`: Kiểm chứng khả năng khởi tạo, đọc, ghi và sửa file `User.md` (đính chính thông tin từ Đà Nẵng sang Huế).
- `test_compact_trigger`: Kiểm chứng cơ chế tự động nén lịch sử và tạo bản tóm tắt khi tổng token vượt ngưỡng.
- `test_cross_session_recall`: Kiểm chứng `AdvancedAgent` nhớ xuyên suốt qua các session mới, trong khi `BaselineAgent` hoàn toàn quên.
- `test_compact_reduces_prompt_load_on_long_thread`: Kiểm chứng lượng prompt context của `AdvancedAgent` thấp hơn đáng kể so với `BaselineAgent` trên hội thoại dài.

---

## 3. Kết quả Benchmark Thực nghiệm

Benchmark được thực hiện trên 2 bộ dữ liệu chuẩn tiếng Việt tại thư mục `data/`:

### Bảng 1: Standard Benchmark (`data/conversations.json` - 10 hội thoại)
*Mục tiêu: Đánh giá khả năng ghi nhớ thông tin người dùng qua nhiều phiên hội thoại thông thường.*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 2,677 | 18,071 | **0.0%** | 0.10 | 0 | 0 |
| **Advanced Agent** | 3,184 | 26,829 | **100.0%** | **1.00** | 253 | 0 |

---

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json` - Hội thoại dài 16 lượt)
*Mục tiêu: Làm lộ rõ chi phí ngữ cảnh tích lũy của Baseline và chứng minh hiệu quả giảm tải của Compact Memory.*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 463 | **22,998** | **0.0%** | 0.10 | 0 | 0 |
| **Advanced Agent** | 894 | **10,220** | **100.0%** | **1.00** | 206 | **5** |

---

## 4. Phân tích Chi tiết & Đánh đổi (Trade-off Analysis)

### 4.1. Vì sao Advanced Agent có Recall vượt trội (100% so với 0% của Baseline)?
- **Baseline Agent**: Lưu trữ trạng thái phiên theo từng `thread_id` độc lập trong RAM. Khi chuyển sang thread kiểm tra recall (`recall_thread_id`), thread mới hoàn toàn trống rỗng $\rightarrow$ Recall bằng 0%.
- **Advanced Agent**: Sử dụng bộ trích xuất thông tin `extract_profile_updates()` để ghi nhận các facts ổn định vào `User.md`. Bất kể câu hỏi recall được đặt ra trong thread nào, agent đều có thể đọc lại hồ sơ từ `User.md` $\rightarrow$ Đạt độ chính xác 100%.

### 4.2. Vì sao Advanced Agent tốn nhiều token hơn ở hội thoại ngắn?
- Trong bảng Standard Benchmark, `Advanced Agent` tiêu tốn **26,829 prompt tokens**, cao hơn mức **18,071** của `Baseline Agent`.
- **Nguyên nhân**: Ở mỗi lượt trả lời, Advanced Agent phải nạp thêm toàn bộ nội dung file `User.md` vào prompt context để agent nắm được thông tin người dùng. Với các hội thoại ngắn (10 lượt), chi phí overhead nạp profile này vượt quá chi phí lưu trữ lịch sử thô. Đây là sự đánh đổi tất yếu (trade-off) để đổi lấy khả năng nhớ dài hạn.

### 4.3. Vì sao Compact Memory giúp Advanced Agent thắng thế ở hội thoại dài?
- Trong Stress Benchmark (16 lượt trao đổi dài), lượng prompt context của `Baseline Agent` tăng theo cấp số nhân ($O(N^2)$), đạt mức **22,998 tokens** do phải kéo theo toàn bộ lịch sử không giới hạn.
- Với `Advanced Agent`, khi tổng token vượt ngưỡng `compact_threshold_tokens` (800 tokens), `CompactMemoryManager` đã kích hoạt nén **5 lần**:
  - Di chuyển các tin nhắn cũ hơn thành một đoạn tóm tắt ngắn (`summary` ~30-50 tokens).
  - Chỉ giữ lại số tin nhắn gần nhất (`keep_messages` = 4 tin nhắn).
- Nhờ đó, lượng prompt context xử lý của Advanced Agent chỉ còn **10,220 tokens** (**giảm hơn 55.5% chi phí ngữ cảnh** so với Baseline) mà vẫn duy trì Recall 100%.

### 4.4. Tăng trưởng File Bộ nhớ (`Memory growth`) và Rủi ro đi kèm
- Kích thước file `User.md` chỉ tăng trưởng ổn định ở mức **206 – 253 bytes**.
- **Rủi ro tiềm ẩn trong thực tế**:
  1. *Lưu nhầm thông tin nhiễu / thông tin tạm thời*: Người dùng nói đùa hoặc nhắc đến địa điểm du lịch tạm thời. Nếu lưu bừa bãi, file `User.md` sẽ bị ô nhiễm thông tin sai.
  2. *Mâu thuẫn thông tin (Conflict/Outdated facts)*: Khi người dùng đổi nghề nghiệp hoặc chuyển nơi ở, nếu hệ thống chỉ chèn thêm dòng mới mà không cập nhật fact cũ, agent sẽ trả lời mâu thuẫn.
  3. *Phình to file theo thời gian*: Cần có cơ chế định kỳ chuẩn hóa và loại bỏ các facts ít được sử dụng.

---

## 5. Các Mở rộng Kỹ thuật 
Hệ thống đã triển khai các cơ chế nâng cao giải quyết các rủi ro trên:

1. **Confidence Threshold & Lọc câu hỏi**:
   - Trong `extract_profile_updates()`, hệ thống tự động phát hiện và bỏ qua các câu hỏi thăm dò (kết thúc bằng dấu `?` hoặc dạng `"bạn có biết..."`, `"bạn thử nhớ..."`), chỉ trích xuất khi người dùng đưa ra câu khẳng định rõ ràng về bản thân.
2. **Lọc thông tin gây nhiễu (Noise Filtering)**:
   - Tự động phát hiện và bỏ qua các câu nói đùa (ví dụ: *"chuyển sang product manager... chỉ là câu đùa"* $\rightarrow$ không đổi nghề sang PM).
   - Nhận diện các địa điểm tạm thời (ví dụ: *"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày"* $\rightarrow$ không cập nhật nơi ở thành Hà Nội).
3. **Xử lý Đính chính & Xung đột Thông tin (Conflict Handling)**:
   - Khi phát hiện đính chính nơi ở (từ Đà Nẵng sang Huế) hoặc nghề nghiệp (từ backend sang MLOps), phương thức `upsert_facts()` ghi đè trực tiếp lên fact cũ trong `User.md`, đảm bảo agent không bao giờ giữ cùng lúc hai thông tin mâu thuẫn.
4. **Hỗ trợ Song song Chế độ Offline & Live LLM (OpenRouter/LangGraph)**:
   - Chế độ **Offline Deterministic**: Phục vụ benchmark nhanh (0.04s), lặp lại được 100% không tốn tiền API.
   - Chế độ **Live LLM**: Tích hợp LangGraph `create_react_agent`, kết nối thành công với các model thực tế qua OpenRouter (`openai/gpt-4o-mini`).

---

## 6. Hướng dẫn Chạy Kiểm thử & Đánh giá

1. **Chạy toàn bộ bài test Pytest**:
   ```bash
   pytest src/test_agents.py -v
   ```
2. **Chạy Benchmark tiêu chuẩn (Chế độ Offline - Nhanh & Ổn định)**:
   ```bash
   python src/benchmark.py
   ```
3. **Kiểm tra kết nối Live LLM với OpenRouter**:
   ```bash
   python src/test_live.py
   ```
4. **Chạy Benchmark với Live LLM (Tùy chọn)**:
   ```bash
   python src/benchmark.py --live --limit 2
   ```
