# Install once before running:
# uv pip install -r asr/requirements.txt
# uv pip install "utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git"

CUDA_VISIBLE_DEVICES=7 uv run python benchmark/run_dataset_benchmark.py \
  --checkpoint_dir checkpoints/vieneu-merged \
  --hf_dataset linhtran92/viet_bud500 \
  --hf_split validation \
  --hf_text_col transcription \
  --sample_limit 100 \
  --prompt_speech_path ref2.wav \
  --text_normalizer none \
  --output_root outputs/benchmarks
