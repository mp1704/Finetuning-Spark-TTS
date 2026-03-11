from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.common import write_json


def _validation_parquet_urls(repo_id: str, split_name: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Run: uv pip install huggingface_hub"
        ) from exc

    split_prefix = f"data/{split_name}-"
    repo_files = list_repo_files(repo_id, repo_type="dataset")
    parquet_paths = sorted(
        path for path in repo_files
        if path.startswith(split_prefix) and path.endswith(".parquet")
    )
    if not parquet_paths:
        raise ValueError(
            f"No parquet files found for split {split_name!r} in dataset {repo_id!r}."
        )

    return [
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
        for path in parquet_paths
    ]


def _resolve_sample_id(sample: dict[str, Any], id_col: str | None, index: int) -> str:
    if id_col:
        value = sample.get(id_col)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"sample_{index:06d}"


def load_hf_validation_samples(
    hf_dataset: str,
    hf_split: str = "validation",
    text_col: str = "transcription",
    id_col: str | None = None,
    sample_limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is not installed. Run: uv pip install datasets"
        ) from exc

    parquet_urls = _validation_parquet_urls(hf_dataset, hf_split)
    dataset = load_dataset(
        "parquet",
        data_files={hf_split: parquet_urls},
        split=hf_split,
    )
    keep_columns = [text_col]
    if id_col:
        keep_columns.append(id_col)
    drop_columns = [column for column in dataset.column_names if column not in keep_columns]
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)
    if sample_limit is not None:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))

    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        raw_text = sample.get(text_col)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        samples.append(
            {
                "id": _resolve_sample_id(sample, id_col, index),
                "text": raw_text.strip(),
                "dataset": hf_dataset,
                "split": hf_split,
                "source_index": index,
            }
        )

    if not samples:
        raise ValueError(
            f"No valid samples found in {hf_dataset}:{hf_split} using text column {text_col!r}."
        )
    return samples


def write_hf_manifest(path: Path, samples: list[dict[str, Any]]) -> None:
    write_json(path, samples)
