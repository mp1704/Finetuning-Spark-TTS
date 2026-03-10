# Chuẩn bị Dataset cho SparkTTS Fine-tuning

Hướng dẫn này giải thích cách chuyển dữ liệu âm thanh thô thành định dạng mà pipeline fine-tuning sử dụng.

---

## Tổng quan pipeline

```
[Bước 0]  Dữ liệu thô: file audio (.wav) + transcript (text)
              ↓
[Bước 1]  Cài dependencies
              ↓
[Bước 2]  Chạy finetune/preprocess.py
          → BiCodec tokenizer đọc từng file audio, tách thành:
              • global_tokens   – đặc trưng giọng người nói (speaker identity)
              • semantic_tokens – nội dung âm thanh (nói gì)
          → Ghi ra file JSONL (train_tokens.jsonl / val_tokens.jsonl)
              ↓
[Bước 3]  finetune/dataset.py đọc JSONL → PyTorch tensors → train.py
```

---

## Yêu cầu dữ liệu thô

| Tiêu chí | Chi tiết |
|---|---|
| **Format audio** | WAV (khuyến nghị), MP3/FLAC cũng được (qua soundfile) |
| **Kênh** | Mono; nếu stereo sẽ tự lấy trung bình 2 kênh |
| **Sample rate** | Bất kỳ – tự resample về 16 kHz bằng soxr |
| **Độ dài clip** | 3–15 giây/clip cho kết quả tốt nhất |
| **Transcript** | Văn bản thuần túy, đúng chính tả, không ký tự đặc biệt lạ |
| **Số lượng** | Tối thiểu ~100 câu để thấy hiệu quả; lý tưởng > 1 000 câu |

---

## Cài đặt dependencies

```bash
uv pip install soxr soundfile

# Nếu dùng HuggingFace dataset:
uv pip install datasets

# Nếu dữ liệu tiếng Việt (cần chuẩn hóa số, ngày tháng, viết tắt):
uv pip install vinorm
```

---

## Cách A – Dataset local (LJSpeech format)

### Cấu trúc thư mục bắt buộc

```
MyDataset/
├── wavs/
│   ├── clip001.wav
│   ├── clip002.wav
│   └── ...
├── metadata_train.csv      ← tập huấn luyện
└── metadata_val.csv        ← tập đánh giá
```

### Định dạng file CSV

Mỗi dòng gồm 3 cột phân cách bằng dấu `|`, **không có header**:

```
clip001|xin chào tôi là trợ lý AI|xin chào tôi là trợ lý AI
clip002|hôm nay 25/12 trời đẹp|hôm nay hai mươi lăm tháng mười hai trời đẹp
```

- Cột 1: ID clip (phải trùng với tên file trong `wavs/`)
- Cột 2: transcript gốc
- Cột 3: transcript đã chuẩn hóa (**cột này được dùng để train**)

> Nếu không cần chuẩn hóa, cột 2 và 3 giống nhau cũng được.

### Lệnh chạy preprocess

```bash
uv run python -m finetune.preprocess \
    --source ljspeech \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --data_dir MyDataset \
    --device cuda
```

Kết quả: `MyDataset/train_tokens.jsonl` và `MyDataset/val_tokens.jsonl`.

---

## Cách B – HuggingFace dataset

Dùng khi dataset đã có sẵn trên HuggingFace Hub với cột audio và cột text.

### Ví dụ 1: ngochuyen_voice (tiếng Việt)

```bash
uv run python -m finetune.preprocess \
    --source huggingface \
    --hf_dataset pnnbao-ump/ngochuyen_voice \
    --hf_audio_col audio \
    --hf_text_col transcription \
    --hf_id_col file_name \
    --text_normalizer vinorm \
    --output_dir data/ngochuyen \
    --val_size 500 \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --device cuda
```

### Ví dụ 2: VieNeu-TTS-140h (tiếng Việt, 140 giờ)

```bash
uv run python -m finetune.preprocess \
    --source huggingface \
    --hf_dataset pnnbao-ump/VieNeu-TTS-140h \
    --hf_audio_col audio \
    --hf_text_col text \
    --hf_id_col _id \
    --text_normalizer vinorm \
    --output_dir data/vieneu \
    --val_size 1000 \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --device cuda
```

### Giải thích các flag quan trọng

| Flag | Ý nghĩa |
|---|---|
| `--hf_audio_col` | Tên cột chứa audio trong dataset |
| `--hf_text_col` | Tên cột chứa transcript |
| `--hf_id_col` | Tên cột dùng làm ID (mỗi dataset khác nhau) |
| `--val_size` | Số mẫu giữ lại cho validation (lấy từ cuối dataset) |
| `--text_normalizer vinorm` | Chuẩn hóa tiếng Việt: số → chữ, ngày tháng, viết tắt |
| `--output_dir` | Thư mục lưu file JSONL (tự tạo nếu chưa có) |

> **Khi nào dùng `--text_normalizer vinorm`?**
> Dùng khi transcript tiếng Việt chứa số, ngày, ký hiệu đơn vị (vd: "25/12", "100km", "TP.HCM").
> Nếu transcript đã viết đầy đủ bằng chữ thì không cần.

---

## File JSONL output

Mỗi dòng trong file JSONL là một JSON record đại diện cho 1 câu:

```json
{
  "id": "clip001",
  "text": "xin chào tôi là trợ lý AI",
  "global_tokens": [12, 45, 7, 3],
  "semantic_tokens": [301, 88, 204, 17, 99, 512, 76, ...]
}
```

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | string | ID định danh clip |
| `text` | string | Transcript (đã chuẩn hóa nếu dùng vinorm) |
| `global_tokens` | list[int] | Đặc trưng giọng người nói, thường 4–8 token, giá trị 0–767 |
| `semantic_tokens` | list[int] | Nội dung âm thanh, độ dài tỉ lệ với thời lượng clip, giá trị 0–1023 |

---

## Cách dataset.py chuyển JSONL thành input cho LLM

Model SparkTTS là một causal language model. Mỗi sample được format thành chuỗi token như sau:

```
INPUT  (không tính loss):
  <|task_tts|>
  <|start_content|> xin chào tôi là trợ lý AI <|end_content|>
  <|start_global_token|> <|bicodec_global_12|><|bicodec_global_45|>... <|end_global_token|>
  <|start_semantic_token|>

TARGET (tính loss, model phải dự đoán):
  <|bicodec_semantic_301|><|bicodec_semantic_88|><|bicodec_semantic_204|>...<eos>
```

- Phần **INPUT** được mask (label = -100), model không học từ đây.
- Phần **TARGET** là các semantic token, model học cách sinh ra chúng.
- Nếu chuỗi quá dài (> `max_length=2048`), semantic token bị cắt từ cuối.
- Khi batch nhiều sample, các chuỗi ngắn hơn được padding về độ dài lớn nhất trong batch.

---

## Cấu hình config.yaml

Sau khi có JSONL, chỉ cần sửa 2 dòng trong [finetune/config.yaml](config.yaml):

```yaml
data:
  train_jsonl: data/ngochuyen/train_tokens.jsonl   # ← đổi thành đường dẫn của bạn
  val_jsonl:   data/ngochuyen/val_tokens.jsonl     # ← đổi thành đường dẫn của bạn
  max_length:  2048
```

---

## Kiểm tra nhanh trước khi train

Chạy đoạn Python sau để verify file JSONL hợp lệ:

```python
import json

path = "data/ngochuyen/train_tokens.jsonl"
with open(path) as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"Tổng số mẫu: {len(records)}")
print(f"Ví dụ: {records[0]}")
print(f"Độ dài semantic_tokens trung bình: {sum(len(r['semantic_tokens']) for r in records) / len(records):.0f} token")
```

Kết quả mong đợi: độ dài semantic trung bình khoảng 50–400 token (tùy độ dài clip).
