import json
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Resolve project root from current working directory (works in notebooks).
def _find_project_root() -> Path:
    markers = [
        Path("pretrained_models/Spark-TTS-0.5B/config.yaml"),
        Path("sparktts/models/audio_tokenizer.py"),
    ]
    for base in [Path.cwd(), *Path.cwd().parents]:
        if all((base / m).exists() for m in markers):
            return base
    # Fallback for normal repo layout when running from nbs/
    return Path.cwd().resolve().parent


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# Notebook-style quick inspect: read 1 line from val_tokens.jsonl
jsonl_path = PROJECT_ROOT / "data/ngochuyen/val_tokens.jsonl"
line_no = 1  # change this (1-indexed)

with jsonl_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        if i == line_no:
            row = json.loads(line)
            break
    else:
        raise ValueError(f"Line {line_no} not found in {jsonl_path}")

print("id:", row["id"])
print("text:", row["text"])
print("global_tokens_len:", len(row["global_tokens"]))
print("semantic_tokens_len:", len(row["semantic_tokens"]))

# show token values
print("global_tokens:", row["global_tokens"])
print("semantic_tokens:", row["semantic_tokens"][:120], "...")  # preview first 120

# Decode to word/text tokens (LLM tokenizer)
text_tokenizer = AutoTokenizer.from_pretrained(
    str(PROJECT_ROOT / "pretrained_models/Spark-TTS-0.5B/LLM")
)
text_ids = text_tokenizer.encode(row["text"], add_special_tokens=False)
text_pieces = [
    text_tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    for tid in text_ids
]

print("\ntext_token_ids:", text_ids)
print("text_token_pieces:", text_pieces)

# NOTE:
# - global_tokens / semantic_tokens are audio codec token IDs, not word tokens.
# - They cannot be decoded directly to words.
# - If needed, we can show them as symbolic tokens like <|g_123|>, <|s_456|>.

# %%
# "Decode" global tokens to symbolic token string used by SparkTTS LLM
global_symbolic = "".join(f"<|bicodec_global_{i}|>" for i in row["global_tokens"])
print("\nglobal_symbolic_preview:", global_symbolic[:240], "...")

# Round-trip via tokenizer (string -> ids -> string) for inspection
global_tok_ids = text_tokenizer.encode(global_symbolic, add_special_tokens=False)
global_tok_text = text_tokenizer.decode(
    global_tok_ids, clean_up_tokenization_spaces=False
)
global_back_to_int = [int(x) for x in re.findall(r"bicodec_global_(\d+)", global_tok_text)]

print("global_symbolic_tokenizer_ids_len:", len(global_tok_ids))
print("global_roundtrip_ok:", global_back_to_int == row["global_tokens"])
print("global_back_to_int_preview:", global_back_to_int[:20], "...")

#%%