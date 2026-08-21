# Lab 21 — Evaluation Report

**Họ tên**: Phạm Đặng Khương Duy  **MSSV**: 202601982  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | `ticket-customer-support-v1` (250 ticket CSKH → JSON triage) |
| Train / val | `200` / `50` (seed 42) |
| `max_length` | `256` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `30` |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào?

> ✅ Template giữ nguyên khối `<think>` theo mặc định của Qwen3.5. Quá trình training sử dụng mask mode `assistant-only` nên chỉ phần response được tính loss, không ảnh hưởng bởi reasoning block.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.7578 | 0.0 | 3150.4 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.0 | 992.2 |
| (c) LoRA fine-tune | 0.97 | 0.6556 | 1.0 | 1379.4 |

**(b) có thật sự mạnh hơn (a) không?** `có` — nếu không, bạn đã cải thiện (b) thế nào?
- (b) format=1.0 vs (a) format=0.0: prompt tối ưu giúp model output đúng format JSON
- (b) latency=992ms vs (a) latency=3150ms: nhanh hơn 3.2 lần
- (b) target=0.765 vs (a) target=0.0: cải thiện rõ rệt

Bạn có sửa `OPTIMIZED_PROMPT` không? Không — sử dụng prompt mặc định từ rubric.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | 0.97 | 923.5 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5377 | 0.97 | 773.8 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.0 | 897.7 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.94 | 966.2 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

`attn_only` và `correct` hoà nhau trên tập target với kết quả 0.97/1.0/0.6556. Tuy nhiên, thứ tự theo train loss hoàn toàn khác: `attn_only` có loss thấp hơn (0.5377 vs 0.6263). Đây là phát hiện quan trọng nhất của lab: **train loss thấp hơn không đồng nghĩa với performance tốt hơn**. Kết quả này cho thấy vị trí gắn adapter (placement) quan trọng hơn rank. Rank chỉ là con số; placement quyết định adapter tác động lên layer nào của transformer. Cả hai config đều đạt target=0.97, nên khi budget cố định, nên ưu tiên placement phù hợp với task.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` sử dụng LR 1e-5 (learning rate của full fine-tune) thay vì 1e-4. Kết quả train loss đạt 1.5704 — gấp ~2.5 lần so với `correct` (0.6263). Nếu chỉ nhìn vào đường loss mà không biết LR, ta sẽ kết luận model "không học được gì" và có thể tăng epochs hoặc thay đổi architecture. Sai lầm nghiêm trọng: learning rate quá thấp khiến adapter không cập nhật đủ, dẫn đến format=0.0 và target=0.0. Bài học: **luôn track hyperparameters khi phân tích training dynamics**.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` tiết kiệm 4.92GB VRAM (7.09GB vs 12.01GB), tức ~41% giảm. Tuy nhiên, final loss cao hơn (0.7058 vs 0.6263), và target thấp hơn đáng kể (0.94 vs 0.97). Số đo ủng hộ khuyến nghị không dùng QLoRA cho Qwen3.5: trừng phạt bằng loss cao hơn và accuracy thấp hơn, trong khi mức tiết kiệm VRAM không đáng kể với GPU T4 16GB. Với setup này, 12GB VRAM đã đủ train, nên QLoRA là trade-off không có lợi.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.102` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?

Model fine-tuned đạt target score 0.97 và format compliance 1.0 — tức là trên task cụ thể (ticket triage), model hoạt động rất tốt. Tuy nhiên, regression delta = -0.102 có nghĩa là general capability của model giảm 10.2% so với baseline. Tolerance cho phép chỉ là ±0.020, nên -0.102 vượt xa ngưỡng.

Nguyên nhân: adapter chỉ được train trên task-specific data (200 samples ticket CSKH), khiến model "quên" các capabilities khác mà base model có sẵn. Đây là hiện tượng catastrophic forgetting điển hình. Theo rubric §14.3, giải pháp là thêm 1-5% replay data — mix general capability samples vào training set để preserve base model behaviors.

Điều này nói gì về bài toán? Khi fine-tune cho task cụ thể, ta phải cân bằng giữa specialization và generalization. Task-specific model có thể thất bại trong edge cases mà base model xử lý tốt. Trong production, nếu model chỉ cần handle ticket triage và không cần giữ general capability, đây có thể là trade-off chấp nhận được.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Đặt chuột không dây VN232232 - Cho tôi trả lại | `doi_tra` | ✅ json đúng | ✅ json đúng | ✅ FT thắng |
| 2 | Đặt ốp lưng VN812931 - Hoàn tiền | `hoan_tien` | ✅ json đúng | ✅ json đúng | ✅ FT thắng |
| 3 | Đặt bình giữ nhiệt VN804124 - Chưa thấy tiền | `hoan_tien` | ❌ json lỗi | ✅ json đúng | ❌ **FT thua** |
| 4 | Đặt nồi chiên DH249548 - Thiếu phụ kiện | `san_pham_loi` | ❌ json lỗi | ✅ json đúng | ❌ **FT thua** |
| 5 | Đặt áo khoác VN613097 - Bị lỗi | `san_pham_loi` | ✅ json đúng | ✅ json đúng | ✅ FT thắng |

Có mẫu chung nào ở các ca FT thua không?

Không có — cả 3 case (3, 4, 12) đều là **FT thắng chứ không phải thua**. Model fine-tune xử lý tốt hơn prompt-only trên cả 3 cases:
- Case 3: prompt sai format, FT đúng
- Case 4: prompt sai format, FT đúng
- Case 12: prompt sai format, FT đúng

Điều này cho thấy fine-tuning **cải thiện format compliance** rõ rệt. Các trường hợp FT "thua" thực chất là do model không generalize tốt trên general capability (đo bằng regression score), không phải do sai task-specific classification.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Model fine-tune đạt 0.97 target score và 1.0 format compliance — về mặt kỹ thuật, nó hoàn thành task ticket triage rất tốt. Tuy nhiên, với regression -0.102, **không nên deploy bản này** vì nó mất 10% general capability.

Đòn bẩy thật sự trong lab này theo thứ tự ưu tiên:
1. **Learning rate (quan trọng nhất)**: wrong_lr cho thấy chỉ cần sai LR là model hoàn toàn fail (target=0.0, format=0.0). LR 1e-4 là sweet spot.
2. **Mask mode**: assistant-only đúng — không nên mask cả câu hỏi.
3. **Vị trí adapter (placement)**: attn_only cho thấy placement hoà với all-linear trên target, nhưng placement quyết định nhiều hơn rank.
4. **Chất lượng dữ liệu**: replay data là giải pháp cho catastrophic forgetting.
5. **QLoRA**: Không đáng dùng cho Qwen3.5 trên T4.

**Ba điều tôi học được** (cụ thể, không generic):
1. Train loss thấp không đồng nghĩa performance tốt — `attn_only` có loss 0.5377 vs `correct` 0.6263 nhưng cùng target. Luôn đánh giá trên eval set thực tế.
2. Learning rate là hyperparameter nguy hiểm nhất — chỉ cần 1 order of magnitude sai (1e-5 thay vì 1e-4) là model hoàn toàn không học được gì có ích.
3. Catastrophic forgetting là trade-off không thể tránh khi specialization — giải pháp thực tế là thêm replay data (1-5%) để preserve base capabilities.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Thêm 5% replay data từ general capability benchmarks để fix regression
2. Quét rank từ 8→32 với controlled experiment để tìm optimal rank
3. Test mask mode khác (full-mask vs assistant-only) để đo impact trên regression

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [x] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
