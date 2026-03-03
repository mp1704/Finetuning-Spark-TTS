"""
PyTorch Dataset for SparkTTS LLM finetuning on tokenized LJSpeech data.

Each sample is a causal-LM sequence:
  INPUT  (loss masked): <|task_tts|><|start_content|>TEXT<|end_content|>
                        <|start_global_token|>GLOBAL_TOKENS<|end_global_token|>
                        <|start_semantic_token|>
  TARGET (loss active): SEMANTIC_TOKENS + <eos>
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class LJSpeechTTSDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            jsonl_path: Path to preprocessed JSONL (output of finetune/preprocess.py).
            tokenizer:  HuggingFace tokenizer for the SparkTTS LLM.
            max_length: Sequences longer than this are truncated from the right (semantic tokens).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_jsonl(jsonl_path)

    def _load_jsonl(self, path: str) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _build_prompt(self, text: str, global_tokens: List[int]) -> str:
        global_str = "".join(f"<|bicodec_global_{i}|>" for i in global_tokens)
        return (
            "<|task_tts|>"
            "<|start_content|>" + text + "<|end_content|>"
            "<|start_global_token|>" + global_str + "<|end_global_token|>"
            "<|start_semantic_token|>"
        )

    def _build_target(self, semantic_tokens: List[int]) -> str:
        return "".join(f"<|bicodec_semantic_{i}|>" for i in semantic_tokens)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        prompt_str = self._build_prompt(item["text"], item["global_tokens"])
        target_str = self._build_target(item["semantic_tokens"])

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)

        # Append EOS after the semantic tokens
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        # Combine and truncate to max_length (trim semantic tail if too long)
        full_ids = prompt_ids + target_ids
        if len(full_ids) > self.max_length:
            # Keep the full prompt; trim semantic tokens from the right
            allowed_target = self.max_length - len(prompt_ids)
            target_ids = target_ids[:max(1, allowed_target)]
            full_ids = prompt_ids + target_ids

        # Labels: -100 for prompt portion, real ids for target
        labels = [-100] * len(prompt_ids) + target_ids

        assert len(full_ids) == len(labels)

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad sequences to the max length in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list, labels_list, attention_mask_list = [], [], []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }
