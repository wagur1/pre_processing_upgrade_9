# RESULTS — v9 per-codec POST (shared trunk + codec FiLM)

## Kết quả (shards 0+1 = 770/1159 seqs, 5k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep (PRE-v1) | −1.51% | [−3.88, +0.96] | −1.53% | [−3.38, +0.33] | 0.892 / 0.945 |
| sandwich (per-codec POST) | −1.76% | [−4.30, +1.01] | −2.90% | [−4.70, −0.93] | 0.896 / 0.998 |

## Verdict: KHÔNG giữ được kỷ lục nào (downside scenario ~25% đã đăng ký)

- h264 −1.76 so với kỷ lục E2 **−5.89** (cùng warm-start!) — POST sau 1079 bước
  alternating STE **mất** phần lớn giá trị h264 của chính nó
- h265 −2.90 so với kỷ lục STE **−3.56**
- Gates cho biết routing ĐÃ học (FiLM 0→0.151, embeddings phân biệt) nhưng
  trunk chia sẻ +continued STE kéo POST về điểm trung bình compromise —
  đúng rủi ro "FiLM conflict" đã đăng ký trong MODEL_PERCODEC.md

## Bài học (bổ sung vào bản đồ falsification)

1. **Specialization đã có là tài sản mong manh**: tiếp tục train trên nhiều
   codec với trunk chung PHÁ huỷ specialization hiện có nhanh hơn là học
   routing mới (−5.89 → −1.76 chỉ sau 1079 bước lr 3e-5).
2. Điều này + E1/E3 xác nhận một nguyên lý: trong regime dữ liệu/budget này,
   **per-codec specialization thắng mọi dạng sharing/co-training** — cấu hình
   tối ưu vẫn là ghép module đã chuyên biệt (frankenstein E2).
3. Nghi vấn mở: E1's POST (tuned x264) thua E2's POST (tuned h265) TRÊN h264 —
   dynamics của STE fine-tune không đơn giản là "chuyên về codec trong vòng lặp".

## Số liệu cuối của toàn chiến dịch (xem v8/docs/RESULTS_sandwich.md)

Best single model: **E2 frankenstein-STE — h264 −5.89% [−7.93,−3.76] /
h265 −2.79% [−4.24,−1.25]** (full n=1159, P=1.000/1.000).
Per-codec records: h264 E2 −5.89 / h265 STE-sandwich −3.56.
