# Install once before running:
# uv pip install -r asr/requirements.txt

CUDA_VISIBLE_DEVICES=7 uv run python benchmark/inference_wer.py \
  --checkpoint_dir checkpoints/vieneu-lora \
  --prompt_speech_path ref2.wav \
  --text_normalizer none \
  --result_csv outputs/result_wer.csv
