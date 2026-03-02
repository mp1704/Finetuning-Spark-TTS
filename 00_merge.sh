CUDA_VISIBLE_DEVICES=7 uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained('pretrained_models/Spark-TTS-0.5B/LLM', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, 'checkpoints/phase1')
model = model.merge_and_unload()
model.save_pretrained('pretrained_models/Spark-TTS-0.5B-vi/LLM')

tok = AutoTokenizer.from_pretrained('pretrained_models/Spark-TTS-0.5B/LLM')
tok.save_pretrained('pretrained_models/Spark-TTS-0.5B-vi/LLM')
print('Done.')
"

# Symlink BiCodec and wav2vec2 (no need to copy large files)
mkdir -p pretrained_models/Spark-TTS-0.5B-vi
ln -sf $(pwd)/pretrained_models/Spark-TTS-0.5B/BiCodec        pretrained_models/Spark-TTS-0.5B-vi/BiCodec
ln -sf $(pwd)/pretrained_models/Spark-TTS-0.5B/wav2vec2-large-xlsr-53  pretrained_models/Spark-TTS-0.5B-vi/wav2vec2-large-xlsr-53
cp pretrained_models/Spark-TTS-0.5B/config.yaml pretrained_models/Spark-TTS-0.5B-vi/config.yaml
