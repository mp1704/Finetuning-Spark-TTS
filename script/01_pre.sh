# CUDA_VISIBLE_DEVICES=7 uv run python -m finetune.preprocess \
#   --source huggingface \
#   --hf_dataset pnnbao-ump/ngochuyen_voice \
#   --hf_audio_col audio \
#   --hf_text_col transcription \
#   --text_normalizer vinorm \
#   --output_dir data/ngochuyen \
#   --val_size 500 \
#   --model_dir pretrained_models/Spark-TTS-0.5B \
#   --device cuda

CUDA_VISIBLE_DEVICES=7 uv run python -m finetune.preprocess \
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
