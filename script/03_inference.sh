# CUDA_VISIBLE_DEVICES=7 uv run python inference/run_finetuned_inference.py \
#   --model_dir pretrained_models/Spark-TTS-0.5B \
#   --adapter_dir checkpoints/ngochuyen-demo/ \
#   --text "đặc biệt, Nhà trường có 05 giảng viên tham gia Đoàn cứu hộ quốc tế sau thảm họa động đất tại Thổ Nhĩ Kỳ." \
#   --prompt_speech_path LJSpeech-1.1/wavs/LJ050-0001.wav \
#   --output_path outputs/ngochuyen_finetuned.wav

CUDA_VISIBLE_DEVICES=7 uv run python inference/run_finetuned_inference.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --adapter_dir checkpoints/ngochuyen-demo/ \
  --text "đặc biệt, Nhà trường có 05 giảng viên tham gia Đoàn cứu hộ quốc tế sau thảm họa động đất tại Thổ Nhĩ Kỳ." \
  --prompt_speech_path audio/ngochuyen_13415.wav \
  --output_path outputs/ngochuyen_finetuned.wav
