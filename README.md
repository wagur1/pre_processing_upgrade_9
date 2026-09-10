# pre_processing_upgrade_9 — PER-CODEC POST: restoration biết codec nó đang vá

**VCM sandwich với POST điều kiện theo codec — thiết kế rút ra từ bằng chứng
8 thí nghiệm của v8** (xem `docs/RESULTS_sandwich.md` bên v8):

| Bằng chứng v8 | Hàm ý thiết kế |
|---|---|
| E2 (POST-STE-h265): h264 **−5.89** | POST có thể mạnh trên codec A |
| STE (in-loop h265): h265 **−3.56** | …và codec B |
| E1 (POST-STE-x264): −4.49/−1.96 | **một POST đơn không thắng cả hai** — STE là trade-off theo codec |
| E3 co-adapt: −3.07/−2.92 | fine-tune nhẹ làm mất chuyên biệt |
| TTO: −3.79/+6.09 | tối ưu per-clip kế thừa bias codec của proxy |

→ **Restorer phải BIẾT mình đang đảo artifact của codec nào.**

## Kiến trúc (`src/models/percodec_sandwich.py`)

```
x ─► PRE (UP-VCM) ─► x264/x265 (đóng băng) ─► decode ─► POST(·, codec, QP) ─► analyzer

POST = 1 trunk UNet dùng chung (basis deblock/dering chung hai codec)
     + codec embedding (8-d) ⊕ QP → FiLM ở bottleneck
     (chỗ codecs KHÁC nhau mới tiêu tốn capacity riêng)
```

- **Warm-start từ 2 kỷ lục v8**: PRE = v7-v1 (−2.46 pre-only) + trunk = STE-POST
  (h265 −3.56). Codec FiLM zero-init ⇒ khởi đầu **đúng bằng** hành vi v8
  (test chứng minh exact) — codec routing học trong STE.
- Train **STE từ đầu** (codec thật trong vòng lặp, engine truyền codec vào
  `post_restore`), 1500 bước calibration, lr 3e-5.
- Dead-saddle guard: out conv noise-init + regression tests (bài học v7/v8).

## Mục tiêu đã đăng ký trước khi chạy

- Center (~50%): h264 ≥ −5.5 / h265 ≥ −3.2 (giữ cả 2 kỷ lục trong 1 model)
- Upside (~25%): h264 −6.5…−7.5 / h265 −3.8…−4.5 (routing cho phép mỗi codec
  phát triển tiếp từ kỷ lục của nó)
- Downside (~25%): routing không tách nổi (FiLM xung đột) → ≈ E1/E2 giữa

## Pipeline (giữ nguyên protocol chuẩn)

```bash
pytest -q                                            # 102 tests
python ops/make_v9_ckpt.py --v1 <v1.pth> --ste <ste.pth> --out outputs/v9_warmstart.pth
# Kaggle:
source ops/kaggle_env.sh <account>
python ops/push_kernel.py train --commit <sha> --config configs/percodec_ar.yaml \
    --init-from <v8-train-kernel> --accelerator NvidiaTeslaT4
python ops/push_kernel.py eval --commit <sha> --shard-idx N --num-shards 3 \
    --train-kernel <account>/u9-train --accelerator NvidiaTeslaT4
python ops/merge_eval.py <shards> --out outputs/eval_v9_full --bootstrap 10000
```

Protocol: held-out `r2plus1d_18`, x264+x265 preset medium QP30–50, n=1159
(fingerprint `30f083f8520a`), BD-Rate + bootstrap CI + gap rule ≥ −0.05.

## Cấu trúc thêm so với v8

```
src/models/percodec_sandwich.py   # PerCodecPostSandwich (codec-FiLM POST)
ops/make_v9_ckpt.py               # lắp warm-start (PRE-v1 + trunk-STE)
configs/percodec_ar.yaml          # STE-from-start, 1500 steps calibration
tests/test_percodec.py            # 6 tests (routing, warm-start exact, guards)
```
