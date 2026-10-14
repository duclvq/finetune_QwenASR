# Báo cáo tổng hợp: tinh chỉnh và triển khai Qwen3-ASR-1.7B cho tiếng Mã Lai và nói trộn Mã Lai – Anh

Tài liệu này tổng hợp toàn bộ công việc đã thực hiện cho bài đánh giá kỹ thuật, từ lựa chọn dữ liệu, tinh chỉnh mô hình, đánh giá, lượng tử hoá, đến triển khai và kiểm tra tải. Mọi số liệu đều tái tạo được từ thư mục `results/malay/`, `outputs/bench_api/` và các script trong kho mã `github.com/duclvq/finetune_QwenASR` (nhánh `main`). Báo cáo chi tiết bằng tiếng Anh nằm ở `docs/malay.md` và `deploy/README.md`; bộ slide trình bày là `docs/assessment_slides.pptx`.

## 1. Mô hình và mục tiêu

Qwen3-ASR-1.7B gồm một bộ mã hoá âm thanh và một bộ giải mã ngôn ngữ Qwen3 1.7 tỷ tham số (tổng cộng 2,06 tỷ). Mô hình sinh ra chuỗi dạng `language Malay<asr_text>...`, trong đó thẻ ngôn ngữ được ép khi suy luận. Mô hình gốc đã nhận dạng tiếng Mã Lai ở mức khá, nên mục tiêu của công việc không phải dạy ngôn ngữ mới mà là xác định mô hình sai ở đâu và dữ liệu nào sửa được lỗi đó, sau đó đưa mô hình tốt nhất vào phục vụ với số luồng đồng thời cao nhất có thể trên một GPU.

Công việc được thực hiện trên ba môi trường: máy tính Windows với GPU RTX 5060 Ti 16 GB (chuẩn bị dữ liệu, toàn bộ các lần huấn luyện, lượng tử hoá, triển khai và kiểm tra tải), Google Colab A100 40 GB (tiếp tục huấn luyện lần chạy cuối từ checkpoint lưu trên Drive), và Docker Desktop trên WSL2 cho phần phục vụ bằng vLLM.

## 2. Dữ liệu

Ba bộ dữ liệu huấn luyện và hai bộ kiểm tra ngoài được dùng.

**Bộ nền tiếng Mã Lai** là `mesolitica/malaya-speech-malay-stt` trên Hugging Face, toàn bộ khoảng 1,63 triệu đoạn, 2.600 giờ, nhãn do Google Speech-to-Text sinh bán giám sát, chữ thường, không dấu câu, lẫn các nguồn web như YouTube và podcast, không gắn nhãn lĩnh vực. Công việc dùng một tập con ngẫu nhiên 50 giờ để huấn luyện và 1.994 đoạn giữ riêng làm tập kiểm tra cùng miền. Khoảng 6% đoạn có nhãn bị cụt so với phần tiếng nói thực tế; các đoạn này được phát hiện bằng Silero VAD và loại bỏ, sau đó bù lại cho đủ 50 giờ.

**Bộ nói trộn và tiếng Anh** gộp ba nguồn: `emhaihsan/Synth-Manglish` (câu Mã Lai trộn tiếng Anh tổng hợp bằng TTS, khoảng 7 giờ, đồng thời cho tập `cs_test` 221 đoạn), 200 đoạn thật cắt từ một tập podcast Malaysia với nhãn máy (chỉ dùng huấn luyện, không báo cáo kết quả), và 5 giờ tiếng Anh đọc từ `google/fleurs` cấu hình `en_us`, gắn thẻ `language English`. Bộ này được thêm sau khi mô hình chỉ học tiếng Mã Lai bắt đầu viết các từ tiếng Anh thành từ Mã Lai.

**BabelSpeech** là `BabelSpeech/50hours_Malay_Real-world_Colloquial_Conversation_and_Monologue_Speech_Dataset` trên Hugging Face (giấy phép Apache 2.0). Bản tải về có 34,9 giờ, 10.319 đoạn, chỉ từ 10 người nói, trong đó người nói spk005 chiếm 23 giờ. Nội dung là tiếng Mã Lai khẩu ngữ từ YouTube, xen tiếng Anh tự nhiên, thuộc các lĩnh vực công nghệ, giải trí, du lịch, giáo dục và đời sống; nhãn được người bản ngữ kiểm tra. Công việc dùng 12 giờ để huấn luyện với trần 10 giờ mỗi người nói, và giữ nguyên người nói spk001 (1,8 giờ, 377 đoạn) làm tập kiểm tra nói trộn thật `babel_test` mà mô hình chưa từng nghe.

**Tập kiểm tra ngoài miền** là `Revolab/ASR-Benchmark-Public`, 820 đoạn có nhãn người trong 12 lĩnh vực (tin tức, nghị viện, đọc, Common Voice, FLEURS, kịch, hoạt hình, podcast, phỏng vấn đường phố, điện thoại, hát, câu ngắn), chấm bằng bộ chuẩn hoá văn bản của chính benchmark.

Mọi bộ dữ liệu đi qua cùng một bước tiền xử lý: giải mã, lấy mẫu lại 16 kHz đơn kênh int16, bỏ đoạn dài hơn 30 giây và nhãn rỗng, chữ thường, bỏ dấu câu, rồi ghi mỗi phần thành một tệp âm thanh liên tục kèm chỉ mục JSON để bộ huấn luyện ánh xạ bộ nhớ trực tiếp.

## 3. Phương pháp tinh chỉnh

Phương pháp được cố định một lần để mọi khác biệt về sau đều là khác biệt về dữ liệu: LoRA hạng 16, alpha 32, trên các phép chiếu attention và MLP của bộ giải mã, bộ mã hoá âm thanh đóng băng ở bf16. Số tham số huấn luyện là 17,4 triệu (0,85% mô hình), mỗi adapter nặng 70 MB và gộp lại được vào mô hình gốc để phục vụ. Thiết lập chung: 2 epoch, tốc độ học 1e-4 theo lịch cosine với 3% khởi động, batch hiệu dụng 16, bf16, gradient checkpointing, seed 42, đánh giá mỗi 200 bước trên 200 đoạn kiểm định, checkpoint có trạng thái optimizer nên tiếp tục được sau gián đoạn, kể cả khi đổi máy. LoRA được chọn vì một thí nghiệm dữ liệu chỉ tốn hai đến ba giờ trên GPU 16 GB, điều cho phép thực hiện năm biến thể dữ liệu trong thời gian có.

Năm lần huấn luyện đã thực hiện:

| Lần chạy | Dữ liệu | Bước | GPU đỉnh | Thời gian | Eval loss cuối |
|---|---|---|---|---|---|
| lora | 50 giờ Mã Lai | 3.746 | ~8 GB | 4,1 giờ | 0,777 |
| lora-clean | 50 giờ, bỏ nhãn cụt | 3.778 | 8,4 GB | 1,7 giờ | 0,785 |
| lora-mix | 63 giờ: clean + nói trộn + tiếng Anh | 4.304 | 8,4 GB | 2,3 giờ | 0,785 |
| lora-mix-cont | lora-mix thêm 2 epoch | 4.304 | 8,4 GB | 2,3 giờ | 0,798 |
| lora-babel | 83 giờ: mix + BabelSpeech | 5.098 | 8,75 GB | 0,3 giờ + 3,0 giờ A100 | 0,779 |

Bằng chứng mô hình đã thay đổi và tải lại được: eval loss giảm từ 0,98 xuống 0,78, adapter 70 MB chứa các delta khác không, nạp lại mô hình gốc cộng adapter rồi nhận dạng cho WER cùng miền giảm từ 52 xuống 37, và checkpoint đã gộp nạp được trong transformers, trong HQQ và trong vLLM.

## 4. Kết quả tinh chỉnh

WER / CER (%) trên bốn tập kiểm tra:

| Hệ thống | Cùng miền (1.994) | Revolab (820) | Nói trộn tổng hợp (221) | Nói trộn thật, người nói mới (377) |
|---|---|---|---|---|
| base | 52,19 / 30,61 | 15,01 / 6,99 | 25,69 / 9,77 | 17,94 / 8,69 |
| lora (chỉ Mã Lai) | 37,04 / 21,17 | 16,14 / 8,01 | 28,53 / 13,09 | – |
| lora-clean | 37,31 / 21,12 | 15,76 / 7,53 | 27,70 / 12,25 | – |
| lora-mix | 37,35 / 21,20 | **13,38** / 6,29 | 8,69 / 4,01 | 14,76 / 6,96 |
| **lora-babel** | **37,03** / 21,22 | 13,39 / 6,35 | 8,69 / **3,95** | **7,83** / **4,23** |

Mô hình chỉ học tiếng Mã Lai thắng rõ cùng miền (52 xuống 37) nhưng thua mô hình gốc ngoài miền: trên Revolab 16,1 so với 15,0, riêng lĩnh vực điện thoại từ 14 lên 33, vì các từ tiếng Anh trong câu bị viết thành từ Mã Lai. Việc làm sạch nhãn giảm lỗi xoá từ nhưng không đổi WER tổng vì nhãn tập kiểm tra cũng nhiễu tương đương. Thêm 7 giờ nói trộn tổng hợp và 5 giờ tiếng Anh (lora-mix) khôi phục tiếng Anh và đưa Revolab xuống 13,4, dưới mô hình gốc, với điện thoại từ 38 về 19. Huấn luyện thêm hai epoch trên cùng dữ liệu làm mọi tập xấu đi, cho thấy đường loss phẳng trước đó là do tốc độ học về 0 chứ không phải hội tụ. Thêm 12 giờ hội thoại thật (lora-babel) giảm một nửa lỗi trên người nói chưa từng nghe, từ 14,8 xuống 7,8, tốt hơn ở 277 trong 377 đoạn và kém hơn ở 25 đoạn, trong khi các tập khác không đổi.

Phân tích lỗi cho thấy khoảng một phần ba khoảng cách giữa lora-mix và lora-babel trên tập nói trộn thật là quy ước chính tả của nhãn (nhãn viết "korang" trong khi mô hình viết "kau orang" ở 287 đoạn, kamera/camera, số viết bằng chữ thay vì chữ số). Khi trung hoà các quy ước này ở cả hai phía, lora-mix đạt 11,2 và lora-babel 7,6, tức 3,6 điểm là cải thiện nhận dạng thực. Hai giới hạn cần nêu: tập nói trộn thật chỉ có một người nói từ cùng kho với dữ liệu huấn luyện mới, và mỗi cấu hình chỉ chạy một seed nên chênh lệch một đến hai điểm trên lĩnh vực 50 đoạn nằm trong nhiễu.

## 5. Lượng tử hoá

Adapter tốt nhất được gộp vào mô hình gốc rồi lượng tử hoá trọng số các lớp tuyến tính của bộ giải mã bằng HQQ, không cần dữ liệu hiệu chuẩn, bộ mã hoá âm thanh giữ bf16.

| Độ chính xác trọng số | WER babel_test (377) | WER cùng miền (300) | GPU sau khi nạp |
|---|---|---|---|
| bf16 | 8,62 | 36,22 | 3,80 GB |
| int8, nhóm 64 | 8,65 | 35,90 | 2,66 GB |
| int4, nhóm 64 | 10,06 | 39,42 | 4,66 GB |
| int3 / int2 | 74–145 / 105 | – | 1,9 GB |

Int8 không mất độ chính xác và giảm 30% bộ nhớ trọng số; int4 tốn thêm 1,4 đến 3,2 điểm WER; 3 bit và 2 bit sụp đổ khi không hiệu chuẩn. Trong backend transformers, HQQ giải lượng tử ở mỗi phép nhân ma trận nên chậm hơn bf16 2,7 lần; vì vậy phần phục vụ dùng FP8 của vLLM, vốn không có chi phí này. Không tồn tại định dạng 6 bit thật cho kiến trúc này; kết quả 6 bit trong báo cáo chỉ là mô phỏng làm tròn.

## 6. Triển khai và hiệu năng suy luận

Mô hình được đóng gói trong `deploy/` bằng Docker Compose trên image chính thức `vllm/vllm-openai:v0.14.0` cộng gói `qwen-asr`, phục vụ API tương thích OpenAI tại cổng 8100 với trọng số FP8, 96 chuỗi hoạt động đồng thời, độ dài ngữ cảnh 4.096. Khởi động engine mất 172 giây (chủ yếu là capture CUDA graph), chiếm 11,3 GB VRAM.

**Đường cơ sở (Part 2)** là backend transformers bf16, một đoạn mỗi lần, trên 24 đoạn thuộc ba nhóm độ dài. RTF trung bình 0,39 (p95 0,61), GPU chỉ bận 17% ở 46 W, cho thấy vòng giải mã từng token trong Python là giới hạn chứ không phải GPU. Gộp 8 đoạn trong cùng backend đưa RTF tổng về 0,053.

**Chất lượng qua máy chủ** được chấm trên toàn bộ 377 đoạn babel_test: bf16 qua vLLM 8,56 WER, FP8 qua vLLM 8,85, so với 8,62 ngoại tuyến; FP8 tốn khoảng 0,25 điểm. Một phát hiện kỹ thuật được ghi lại: endpoint nhận dạng của vLLM không đọc giới hạn độ dài sinh từ cấu hình mô hình mà giải mã tới hết 4.096 token, nên hai đoạn khiến bộ giải mã lặp từ đã sinh ra khoảng 1.200 từ mỗi đoạn và đẩy WER cả tập lên 20; mọi yêu cầu vì thế gửi kèm `max_completion_tokens=400`, bằng giới hạn dùng khi đánh giá ngoại tuyến.

**Kiểm tra tải (Part 4)** gửi 256 yêu cầu trọn tệp ở mỗi mức đồng thời:

| Đồng thời | RTF trung bình | RTF p95 | Độ trễ p95 | Thông lượng (lần thời gian thực) | req/s | GPU |
|---|---|---|---|---|---|---|
| 1 | 0,043 | 0,069 | 1,37 s | 24 | 1,9 | 97% |
| 16 | 0,055 | 0,089 | 1,71 s | 294 | 23,3 | 98% |
| 64 | 0,088 | 0,145 | 2,55 s | 614 | 48,7 | 97% |
| 96 | 0,114 | 0,191 | 3,18 s | 685 | 54,4 | 99% |
| 128 | 0,157 | 0,286 | 3,36 s | 684 | 54,3 | 97% |
| 256 | 0,224 | 0,610 | 3,82 s | 674 | 53,4 | 99% |

Một luồng đơn đạt RTF 0,043, nhanh hơn đường cơ sở 9 lần; đoạn 2 đến 5 giây trả lời trong 0,18 giây, đoạn 15 đến 30 giây trong 1,09 giây. Thông lượng bão hoà ở khoảng 685 lần thời gian thực (54 yêu cầu mỗi giây) từ 96 luồng. Số luồng đồng thời duy trì được ở ngưỡng p95 RTF 0,3 là 128; ngưỡng 0,5 bị vượt ở 256. Không yêu cầu nào thất bại ở mọi mức.

**Hành trình tối ưu (Part 3 và 5)** gồm: gộp batch trong transformers (7 lần), HQQ int8 (bị loại cho phục vụ vì chậm 2,7 lần dù tiết kiệm bộ nhớ), chuyển sang vLLM FP8 với continuous batching (cải thiện lớn nhất), quét mức đồng thời tới 256, và giới hạn độ dài sinh. Các thử nghiệm không giúp ích được giữ lại trong hồ sơ: lượng tử 3 và 2 bit, thêm epoch trên cùng dữ liệu, mô hình chỉ học Mã Lai, huấn luyện trên A100 Colab (không nhanh hơn GPU máy bàn vì bộ huấn luyện bị nghẽn CPU ở bước gom batch).

**Điểm nghẽn (Part 6)**: tính toán GPU là giới hạn cuối, với SM bận 97 đến 99% ngay từ 2 luồng, công suất 120 đến 131 W trên trần 150 W (không bị giới hạn công suất), VRAM đứng yên ở 11,3 GB trên 16 GB. Vượt quá 96 chuỗi hoạt động của vLLM, các yêu cầu thừa xếp hàng, đó là lý do p95 tăng từ 0,19 lên 0,61 trong khi thông lượng không đổi; núm điều chỉnh đầu tiên cần thử là nâng số chuỗi hoạt động cùng bộ nhớ KV cache và dùng KV cache FP8. Chạy dưới WSL2 làm vLLM tắt pinned memory, một máy chủ Linux gốc sẽ nhanh hơn.

## 7. Trả lời các câu hỏi cuối

Số luồng đồng thời duy trì được ở p95 RTF không quá 0,5 là ít nhất 128 yêu cầu trọn tệp trên một RTX 5060 Ti 16 GB với FP8; ở ngưỡng 0,3 cũng là 128. Điểm nghẽn chính là tính toán GPU, tiếp đến là trần 96 chuỗi hoạt động của vLLM. Cải thiện lớn nhất là chuyển từ vòng lặp transformers một đoạn sang vLLM FP8 continuous batching, 9 lần trên một luồng và 685 lần thời gian thực khi gộp. Với thêm một tuần, các việc tiếp theo là nâng số chuỗi hoạt động kèm KV cache FP8 và đo chính xác khoảng 128 đến 256, bổ sung luồng streaming với VAD và kết quả tạm thời trên mô hình này, giải mã suy đoán, chuyển sang máy chủ Linux gốc, và lấy vết Nsight của vòng engine. Với 1.000 phiên đồng thời, kiến trúc đề xuất là một tầng nhận kết nối và VAD không trạng thái trước một nhóm bản sao vLLM (khoảng 128 luồng mỗi card 16 GB, tức khoảng 8 card như vậy hoặc 2 đến 3 card lớn hơn), định tuyến theo số chuỗi đang hoạt động, giám sát sức khoẻ và KV cache từng bản sao, và kiểm soát nhận yêu cầu để p95 giảm dần thay vì thất bại.

## 8. Sản phẩm bàn giao

Mã nguồn trên nhánh `main`: bộ huấn luyện `train_qwen_asr.py`, đánh giá `eval_wer.py`, `score_malay.py`, `eval_revolab.py`, lượng tử hoá `eval_quant.py`, các script dựng dữ liệu trong `data_prep/`, chuỗi chạy đầy đủ `scripts/run_malay.sh`, notebook Colab, thư mục `deploy/` với Dockerfile, docker-compose, entrypoint, `bench.py`, `bench_offline.py`, `eval_api.py`, và `plot_malay.py` tái tạo toàn bộ hình. Tài liệu: `README.md`, `docs/malay.md` (báo cáo tinh chỉnh và phục vụ), `deploy/README.md` (triển khai, benchmark, điểm nghẽn), `docs/talk_script_data_and_training.md` (lời dẫn trình bày), `docs/assessment_slides.pptx` (11 slide). Kết quả: `results/malay/` chứa log từng bước của mọi lần huấn luyện, dự đoán từng đoạn của mọi tập kiểm tra, tổng hợp lượng tử hoá và benchmark phục vụ. Adapter của các lần chạy nằm trên máy huấn luyện và Google Drive tại `My_dataset/outputs/<run>/adapter`.
