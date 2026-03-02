#!/bin/bash
# Vietnamese inference with fine-tuned model + reference audio
# Usage: bash example/infer_vi.sh

script_dir=$(dirname "$(realpath "$0")")
root_dir=$(dirname "$script_dir")
cd "$root_dir" || exit

# CUDA_VISIBLE_DEVICES=7 python -m cli.inference \
#     --text "Xin chào" \
#     --save_dir example/results \
#     --model_dir pretrained_models/Spark-TTS-0.5B-vi \
#     --prompt_speech_path example/ref.wav

CUDA_VISIBLE_DEVICES=7 python -m cli.inference \
    --text "Xin chào" \
    --save_dir example/results \
    --model_dir pretrained_models/Spark-TTS-0.5B-vi \
    --gender male \
    --pitch moderate \
    --speed moderate
