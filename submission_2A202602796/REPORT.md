# Báo cáo Lab Day 1 — Hồ Đăng Phúc — 2A202602796

## 1. Thiết lập

- Môi trường: máy cá nhân, GPU NVIDIA GeForce RTX 4050 Laptop, PyTorch 2.11.0+cu128 (in ở ô đầu của notebook).
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val. Chuẩn hoá 10 cột số bằng mean/std của phần train còn lại.
- Model: `M-base` (54→256→128→7, 47 879 tham số, ReLU, không softmax trong model). Baseline: CE, SGD+momentum 0.9, lr = 0.3 (chọn bằng val trong lưới {0.01, 0.03, 0.1, 0.3}), batch 512, 20 epoch, He init, dropout 0, không clip, FP32.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = 0.4876.
- Các chủ đề đã thử: ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | He: 2,207 (trên val); xavier 1,997; default 1,892; normal 1,946; zeros 1,946. He cao hơn ln 7 vì lớp cuối cũng có phương sai 2/n_in nên logit ban đầu lớn |
| Quá khớp 20 mẫu: loss cuối | 2,28 → 1,9e-4 sau 400 bước, accuracy 20 mẫu = 1,00 |
| Mọi tham số có gradient khác 0 | ☑ có (‖grad‖ từ 0,27 đến 1,94 ở cả 6 tensor) |
| Baseline, số seed đã chạy | 5 (`base-s1..s5`) |
| Baseline: val acc (TB ± σ) | 0,9131 ± 0,0011 |
| Baseline: val macro-F1 (TB ± σ) | 0,8595 ± 0,0052 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0,0105 (val macro-F1; từ sheet `Seeds`). Hình: `figures/base-s1.png`; đường cong: train/val loss vẫn đang giảm ở epoch 20 (best epoch = 20 ở s1; 17–20 ở các seed), khoảng cách train–val chỉ ≈ 0,02 → chưa quá khớp, mô hình chưa hội tụ hẳn. Val macro-F1 tăng từ 0,64 (epoch 1) lên 0,86 (epoch 20). Val acc 0,913 cao hơn nhiều mốc đoán đa số 0,4876. Lưu ý σ chỉ từ 5 seed nên khá thô, và baseline dao động khá nhiều ở macro-F1 (lớp hiếm) hơn ở accuracy.

## 3. Kết quả theo chủ đề

> Tất cả dựa trên **val**; Δ so với trung bình baseline 0,8595; ngưỡng 2σ = 0,0105; mỗi cấu hình thí nghiệm chỉ 1 seed.

### 3.1 Hàm mất mát — CE vs MSE
- Dự đoán: MSE học chậm hơn CE, cần lr lớn hơn (≈3–10×) để bắt kịp, vẫn có thể kém hơn về macro-F1.
- Kết quả (`base-s1`, `loss-mse-lr0.3/0.9/3`; ![](figures/compare_loss.png)): CE val macro-F1 0,8595 / acc 0,913. MSE lr 0,3: 0,788 / 0,884 (Δ = −0,072); lr 0,9: 0,706 / 0,837; lr 3: **phân kỳ** (NaN, `diverged = Y`). CE tốt hơn MSE rõ rệt, vượt xa 2σ; MSE hội tụ chậm hơn (train loss 0,026 vs thang CE không so được, nhưng val acc/F1 thấp hơn ở mọi lr thử).
- Giải thích: gradient CE theo logit là `softmax−y`, đủ lớn khi dự đoán sai; gradient MSE là `2(z−y)/(7B)` nhỏ (chia cho 7B) và không có softmax nên tín hiệu học yếu, đặc biệt cho lớp hiếm (macro-F1 thấp). **Khác dự đoán ở chỗ:** tăng lr cho MSE không giúp (lr 0,9 tệ hơn 0,3, lr 3 phân kỳ). Lr tốt nhất của MSE nằm ở mép dưới lưới (0,3), nên chưa biết lr nhỏ hơn có khá hơn không — đây là hạn chế của lưới.

### 3.2 Bộ tối ưu hoá
- Dự đoán: SGD thường cần lr lớn hơn ≈10× SGD+momentum; Adam/AdamW nhỉnh hơn SGD+momentum nhưng chênh lệch có thể nhỏ; lr quá nhỏ (1e-4) thì chưa hội tụ.
- Mỗi bộ ở lr tốt nhất của nó (trong lưới đã thử):

| Bộ tối ưu | exp_id | lr | val macro-F1 | best epoch |
|---|---|---|---|---|
| SGD | `opt-sgd-lr1` | 1 | 0,8409 | 19 |
| SGD+momentum | `opt-sgdm-lr0.3` (= baseline) | 0,3 | 0,8597 | 20 |
| Adam | `opt-adam-lr0.003` | 0,003 | **0,8840** | 20 |
| AdamW (wd = 0,01) | `opt-adamw-lr0.003` | 0,003 | 0,8661 | 19 |

- Độ nhạy với lr: ![](figures/compare_optimizer.png) ![](figures/compare_lr_adam.png) — Adam: 1e-4 → 0,714; 3e-4 → 0,794; 1e-3 → 0,855; **3e-3 → 0,884**; 1e-2 → 0,864 (đỉnh nằm giữa lưới, ổn định trên dải rộng). SGD+momentum: 0,761 / 0,824 / 0,854 / 0,860 (lr 0,01 → 0,3) tăng đều; SGD thuần: 0,736 / 0,788 / 0,841 (lr 0,1 → 1) cũng tăng đều. Lr tốt nhất của cả hai bộ SGD nằm **ở mép trên lưới**, nên kết luận về chúng là cận dưới (lr lớn hơn có thể khá hơn).
- Adam vs AdamW(wd=0) (`opt-adamw-wd0`): trùng nhau hoàn toàn (0,8840 vs 0,8840) — đúng như lý thuyết (AdamW với wd=0 ≡ Adam).
- Giải thích: Adam vượt SGD+momentum 0,0243 (> 2σ), chấp nhận được là thắng ở lr tốt nhất trong lưới vì Adam chia bước theo độ lớn gradient từng tham số nên hội tụ nhanh hơn trong 20 epoch. AdamW (wd=0,01) thấp hơn Adam 0,018 — trên 1 seed khó khẳng định đó là tác dụng thật của weight decay hay nhiễu, nhưng phù hợp với việc mô hình chưa quá khớp nên decay chỉ thêm ràng buộc. SGD thuần với lr 1 ≈ SGD+momentum với lr 0,1 (0,841 vs 0,854), đúng dự đoán momentum 0,9 làm bước hiệu dụng ≈ 10×.

### 3.3 Hyper-parameter
- Yếu tố đã đổi (`hp-*`; ![](figures/compare_hparam.png)):

| exp_id | val macro-F1 | Δ vs base | vượt 2σ? | thời gian/epoch |
|---|---|---|---|---|
| `hp-batch128` (2906 bước/epoch) | 0,7963 | −0,063 | có (xấu hơn) | 5,05 s |
| `hp-batch2048` (182 bước/epoch) | 0,8486 | −0,011 | sát ngưỡng | 0,29 s |
| `hp-batch2048-lr4x` (lr 1,2) | 0,7577 | −0,102 | có (xấu hơn) | 0,28 s |
| `hp-wide` (512-256) | 0,8786 | +0,019 | có | 1,26 s |
| `hp-deep` (256-128-64) | 0,8517 | −0,008 | không | 1,43 s |
| `hp-wd1e-4` | 0,8100 | −0,050 | có (xấu hơn) | 1,81 s |
| `hp-epochs40` | 0,8793 | +0,020 | có | 1,70 s |

- Nhận xét: tăng độ rộng và tăng số epoch đều có ích vượt nhiễu, vì baseline đang chưa hội tụ/chưa khớp. Thêm tầng (deep) không giúp trong 20 epoch. Batch 2048 nhanh gấp ≈4× mỗi epoch nhưng kém hơn vì ít bước cập nhật; tăng lr ×4 theo quy tắc tuyến tính **không bù được** mà còn tệ hơn (lr 1,2 với momentum 0,9 quá lớn). **Khác dự đoán:** batch 128 tệ hơn rõ rệt dù nhiều bước hơn 4× — với cùng lr 0,3, batch nhỏ cho gradient nhiễu hơn nên cần lr nhỏ hơn; và epoch chậm hơn ≈4,4×. Weight decay 1e-4 hại khá nhiều thay vì "ít ảnh hưởng": với mô hình đang underfit, L2 cộng vào gradient (cùng lr 0,3) làm giảm năng lực. Với mô hình đang chưa quá khớp, tăng năng lực/thời gian huấn luyện có lợi, chính quy hoá có hại.

### 3.4 Dropout
- (`drop-*`; ![](figures/compare_dropout.png)) val macro-F1: baseline 0,8595; q=0,1 → 0,8418 (Δ −0,018, vượt 2σ); q=0,3 → 0,7523; q=0,5 → 0,6118. Khoảng cách val−train loss: baseline 0,021; q=0,1 → 0,009; q=0,3 → 0,004; q=0,5 → 0,001.
- Mô hình **không** thực sự quá khớp (khoảng cách chỉ 0,02, val loss vẫn đang giảm ở epoch 20). Dropout làm khoảng cách thu hẹp nhưng bằng cách làm cả train loss tăng (0,200 → 0,418), tức là chỉ làm mô hình underfit hơn. Khớp dự đoán, nhưng q=0,1 cũng hại vượt nhiễu thay vì "trong nhiễu". Dropout chỉ nên dùng khi val loss bắt đầu tăng/khoảng cách train–val lớn.

### 3.5 Gradient clipping
- `grad_norm` trung bình của baseline ≈ 0,35 (trung vị ≈ 0,34) → chọn `c = 0,344` (`clip-c0.34`; tỉ lệ bước bị clip trung bình 0,74, ở các epoch đầu 0,83–0,89).
- Ở lr thường: `clip-c0.34` val macro-F1 0,8521 (Δ −0,007, trong nhiễu). Khác dự đoán ở chỗ tỉ lệ bước bị clip cao (≈ 74%) chứ không phải ≈ 50%, vì `grad_norm` baseline tập trung ngay quanh trung vị nên phân bố rất hẹp.
- Ở lr ×10 = 3,0 (`clip-hilr-none` vs `clip-hilr-c0.34`; ![](figures/compare_clipping.png) ![](figures/compare_clipping_gradnorm.png)): không clip: `grad_norm_max` epoch 1 = 294 (gai gradient), mạng sụp về dự đoán lớp đa số (macro-F1 0,0936, acc 0,4876, grad_norm giảm còn ≈ 0,08), không có NaN; có clip: macro-F1 0,294 ở epoch 1 (tỉ lệ bị clip 42% epoch đầu), cao hơn hẳn không clip nhưng vẫn rất tệ (dao động, sau đó giảm). Clipping làm giảm tác hại của gai gradient nhưng không "cứu" hoàn toàn khi lr vượt xa vùng ổn định: bước tối đa vẫn là lr·c ≈ 1,03.

### 3.6 Mixed precision
- FP32 vs FP16 vs BF16 (`amp-fp16`, `amp-bf16`; ![](figures/compare_amp.png)): thời gian/epoch 1,14 s (FP32) vs 1,96 s (FP16) vs 1,64 s (BF16) — **chậm hơn**; bộ nhớ cực đại 179 / 185 / 185 MB; val macro-F1 0,8597 / 0,8617 / 0,8624 (trong nhiễu, Δ < 2σ). FP16 có 5 bước bị bỏ qua (`skipped_steps` = 5, GradScaler gặp inf/NaN lúc đầu rồi giảm scale), BF16 không có.
- Giải thích: mạng ~48k tham số, batch 512 nên thời gian bị chi phối bởi chi phí gọi kernel chứ không phải số FLOP; autocast thêm các phép ép kiểu và GradScaler thêm bước, nên chậm hơn. Bộ nhớ gần như không đổi vì bị chi phối bởi dữ liệu trên GPU. FP16 cần GradScaler vì khoảng biểu diễn hẹp (gradient nhỏ dễ underflow); BF16 cùng số mũ 8 bit với FP32 nên không cần.

### 3.7 Khởi tạo tham số
- Độ lệch chuẩn kích hoạt theo lớp (ReLU 1 / 2 / logits) ở bước 0 và loss bước 0 (val; ![](figures/compare_init.png)):

| init | std lớp 1 | std lớp 2 | std lớp 3 | loss bước 0 | val macro-F1 |
|---|---|---|---|---|---|
| he (baseline) | 0,274 | 0,113 | 0,059 | 2,207 | 0,8595 |
| default | 0,274 | 0,113 | 0,059 | 1,892 | 0,8591 |
| xavier | 0,276 | 0,218 | 0,192 | 1,997 | 0,8656 |
| normal (σ=0,01) | 0,034 | 0,0038 | 0,00027 | 1,946 | 0,8639 |
| zeros | 0 | 0 | 0 | 1,946 | 0,0936 |

  (Bảng std lấy từ `act_std_step0` trong `results/init-*.json`; "std lớp k" là lớp thứ k trong danh sách đó; mỗi init có loss bước 0 đo trên val khác một chút so với bước 0 của lần huấn luyện.)
- `zeros`: gradient lớp ẩn = 0 vì ReLU(0)=0 và đối xứng → chỉ bias lớp ra học được; mạng chỉ học tần suất lớp (acc 0,4876, macro-F1 0,0936, trùng mốc đoán đa số; `grad_norm` ≈ 0,04 chỉ từ bias lớp ra). `normal`: kích hoạt co rất nhanh theo lớp (0,034 → 0,0038 → 0,00027) nên gradient ban đầu nhỏ, nhưng **vẫn học được** bình thường sau 20 epoch (0,8639, trong nhiễu so với baseline) — khác dự đoán "học chậm". He, default, xavier không khác nhau vượt nhiễu (cả ba trong khoảng 2σ), vì mạng chỉ 3 lớp, chưa đủ sâu để thấy khác biệt.

## 4. Đánh giá cuối trên tập eval

> Số lấy từ `eval_result.json` / output notebook (do `scripts/evaluate.py` tạo).

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (TB 5 seed) | — | 0,8595 | 0,8613 ± 0,0032 | 0,9118 |
| Cấu hình cuối cùng (TB 3 seed) | final-s1 | 0,9061 | 0,9064 ± 0,0008 | 0,9405 |

- Cấu hình cuối cùng gồm: Adam, lr = 0,003, hidden 512-256, 40 epoch, batch 512, He, FP32 — chọn bằng val trong số 3 ứng viên `fin-cand-*`: (a) `fin-cand-opt` Adam 0,003 → 0,8840; (b) `fin-cand-wide` + M-wide → 0,8928; (c) `fin-cand-wide-2x` + 40 epoch → **0,9024** (cao nhất → chọn).
- Cải thiện so với baseline trên eval: Δ = +0,0450 macro-F1 (trên val +0,0465), so với 2σ = 0,0105: **vượt nhiễu** (gấp ≈ 4 lần).
- Val và eval: rất gần nhau (baseline 0,8595 vs 0,8613; cuối 0,9061 vs 0,9064, chênh ≤ 0,002) → val là ước lượng tốt của eval. File nộp `predictions_eval.csv` là của `final-s1`: eval macro-F1 = 0,9073, accuracy = 0,9406. Khoảng cách train–val loss của cấu hình cuối ≈ 0,05 (train 0,114 vs val 0,164), best epoch 38–39/40: bắt đầu có dấu hiệu quá khớp nhẹ.

### 4.1 Phân tích lỗi theo lớp (`final-s1`)

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 Spruce/Fir | 42 368 | 0,9346 | 0,9419 | 0,9382 |
| 1 Lodgepole | 56 661 | 0,9490 | 0,9499 | 0,9494 |
| 2 Ponderosa | 7 151 | 0,9576 | 0,9196 | 0,9382 |
| 3 Cottonwood | 549 | 0,8680 | 0,8506 | 0,8592 |
| 4 Aspen | 1 899 | 0,8948 | 0,7657 | 0,8252 |
| 5 Douglas-fir | 3 473 | 0,8770 | 0,9093 | 0,8928 |
| 6 Krummholz | 4 102 | 0,9418 | 0,9542 | 0,9479 |

- Lớp khó nhất là lớp 4 (Aspen; F1 = 0,8252, recall 0,766), hay bị nhầm với lớp 1 (Lodgepole): 19,1% mẫu Aspen bị dự đoán thành Lodgepole (362/1899). Cặp lớn nhất theo số mẫu là 0↔1 (2231 + 2549), vì hai lớp này lớn và có đặc trưng địa hình/độ cao gần giống nhau. Lớp 3 (Cottonwood, chỉ 549 mẫu) hay bị nhầm với lớp 2 (10,6%).
- Lý giải: lớp 4 chỉ chiếm 1,6% dữ liệu và sống cùng độ cao/loại đất với Lodgepole nên mô hình bị kéo về lớp đa số khi hàm mất mát là CE không có trọng số; lớp 3 cũng hiếm (0,47%) và giống lớp 2. Cách cải thiện: dùng trọng số lớp hoặc focal loss để tăng recall lớp hiếm, huấn luyện lâu hơn/mạng rộng hơn (từ baseline sang cuối đã nâng F1 lớp hiếm), hoặc thêm đặc trưng tương tác (độ cao − khoảng cách thuỷ văn…).

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì sao?** Adam ở lr 3e-3 thắng (0,884 vs SGD+momentum 0,860, vượt 2σ), và Adam ổn định trên dải lr rộng (1e-3 đến 1e-2 đều ≥ 0,855). Khi lr không được chỉnh, kết quả phụ thuộc rất nhiều vào lr: SGD/SGD+momentum ở lr nhỏ kém xa (0,736–0,76), Adam ở lr quá nhỏ 1e-4 cũng kém (0,714). Lưu ý SGD/SGDM tốt nhất nằm ở mép lưới nên có thể được cải thiện nếu thử lr lớn hơn.
2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào nên dùng?** Không giúp mà làm hại (0,8595 → 0,8418 → 0,7523 → 0,6118 khi q = 0 → 0,1 → 0,3 → 0,5) vì mô hình đang underfit (khoảng cách train–val chỉ 0,02). Nên dùng khi val loss bắt đầu tăng/khoảng cách train–val lớn.
3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào chứng minh?** Nó giới hạn độ dài bước khi gradient có gai. Ở lr ×10, `grad_norm_max` epoch 1 là 294 và mạng sụp về lớp đa số (macro-F1 0,094); có clip (c=0,344) macro-F1 là 0,294, tốt hơn nhưng vẫn không hồi phục — clipping chỉ giảm tác hại, không thay thế lr hợp lý. Ở lr thường clipping không đổi gì đáng kể (0,8521, trong nhiễu).
4. **Mixed precision có nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?** Không: chậm hơn (1,64 s BF16, 1,96 s FP16 vs 1,14 s FP32 mỗi epoch), bộ nhớ không giảm (185 vs 179 MB). Mạng nhỏ nên chi phí gọi kernel và autocast/GradScaler lấn át lợi ích hạ độ chính xác; độ chính xác giữ nguyên (trong nhiễu).
5. **Vì sao khởi tạo toàn số 0 hỏng? He khác Xavier ở đâu, khi nào quan trọng?** Với W = 0, kích hoạt ẩn ReLU(0) = 0 nên gradient của W mọi lớp = 0 và các nơ-ron đối xứng; chỉ bias lớp ra học nên mạng chỉ dự đoán lớp đa số (F1 0,094). He dùng Var = 2/n_in (bù việc ReLU bỏ một nửa), Xavier dùng 2/(n_in+n_out) cho kích hoạt đối xứng; trong thực nghiệm này He/Xavier/default không khác nhau vượt nhiễu vì mạng chỉ 3 lớp; sự khác biệt quan trọng khi mạng rất sâu (kích hoạt co/nổ theo độ sâu).
6. **Quay lại câu hỏi của bài học (loss không giảm sau 2 000 bước): 3 phép kiểm tra đầu tiên?** (1) loss bước 0 ≈ ln 7 (khởi tạo/nhãn/chuẩn hoá đúng không) — ở đây 1,89–2,21 với các init thông thường, `zeros` chính xác 1,946; (2) quá khớp được 1 lô 20 mẫu với mọi chính quy hoá tắt (phân biệt lỗi code với thiếu năng lực) — ở đây loss 2,28 → 1,9e-4, acc 1,00; (3) mọi tham số có gradient khác 0 và `grad_norm` (gradient có chảy không) — `init-zeros` có `grad_norm` ≈ 0,04 chỉ từ bias và F1 = 0,094; `clip-hilr-none` có `grad_norm_max` = 294 rồi giảm còn 0,08 cùng F1 = 0,094. Cả hai trường hợp "loss không giảm" đều được phát hiện bằng ba phép kiểm tra này.

## 6. Hạn chế và điều bất ngờ

- Kết quả khác dự đoán: (i) MSE không khá hơn khi tăng lr (lr 0,9 tệ hơn 0,3, lr 3 phân kỳ); (ii) batch 128 tệ hơn rõ rệt dù nhiều bước hơn 4×; (iii) weight decay 1e-4 hại đáng kể thay vì "ít ảnh hưởng"; (iv) clipping ở lr thường kích hoạt ≈ 74% số bước chứ không phải ≈ 50%; (v) init `normal` (σ=0,01) vẫn học bình thường dù kích hoạt co rất mạnh; (vi) clip ở lr ×10 không cứu được mạng.
- Hạn chế thiết kế: 5 seed baseline cho σ thô (σ = 0,0052); cùng số epoch nhưng khác số bước cập nhật; lưới lr thô và lr tốt nhất của SGD, SGD+momentum, MSE nằm ở mép lưới nên chưa chắc đã là tối ưu; mạng chưa hội tụ hẳn sau 20 epoch (best epoch ≈ cuối); chỉ 1 seed cho mỗi cấu hình thí nghiệm nên chênh lệch nhỏ hơn 2σ (ví dụ Adam vs AdamW) không kết luận được.
- Nếu có thêm thời gian: mở rộng lưới lr cho SGD/SGDM/MSE, chạy nhiều seed hơn cho mỗi cấu hình, thử lịch giảm lr (scheduler), và trọng số lớp/focal loss cho lớp hiếm (Aspen, Cottonwood).

## 7. Phụ lục

- File đã nộp: `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/`, `results/`, `code/`.
- Thời gian chạy ước tính tổng cộng: ≈ 26 phút huấn luyện (51 lần chạy) trên RTX 4050 Laptop; ≈ 1,1 s/epoch với cấu hình baseline.
