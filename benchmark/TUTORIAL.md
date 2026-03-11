# Benchmark Tutorial

This folder contains the evaluation workflow for SparkTTS checkpoints.

The recommended dataset benchmark entry point is:

```bash
uv run python benchmark/run_dataset_benchmark.py --help
```

It benchmarks a checkpoint on a Hugging Face validation split and collects:

- WER
- UTMOSv2
- local RTF

The benchmark loader resolves only the requested split parquet files such as
`data/validation-*.parquet`, so it avoids downloading the dataset `train` split.

## Prerequisites

- `uv` installed
- project dependencies installed:

```bash
uv pip install -r requirements.txt
```

- ASR dependencies installed for WER:

```bash
uv pip install -r asr/requirements.txt
```

- UTMOSv2 installed:

```bash
uv pip install "utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git"
```

- Hugging Face access approved for the gated dataset `linhtran92/viet_bud500`
- ASR model downloaded:

```bash
uv run python -m asr.download_model
```

## Main benchmark command

Example with the merged checkpoint in shared-prompt voice cloning mode:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmark/run_dataset_benchmark.py \
  --checkpoint_dir checkpoints/vieneu-merged \
  --hf_dataset linhtran92/viet_bud500 \
  --hf_split validation \
  --hf_text_col transcription \
  --sample_limit 100 \
  --prompt_speech_path ref2.wav \
  --text_normalizer none \
  --output_root outputs/benchmarks
```

Example in controllable mode:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmark/run_dataset_benchmark.py \
  --checkpoint_dir checkpoints/vieneu-merged \
  --hf_dataset linhtran92/viet_bud500 \
  --hf_split validation \
  --hf_text_col transcription \
  --sample_limit 100 \
  --gender female \
  --pitch moderate \
  --speed moderate \
  --text_normalizer none \
  --output_root outputs/benchmarks
```

## Output layout

Each run creates an isolated directory under `outputs/benchmarks/`:

- `samples.json`: benchmark manifest
- `generated_wavs/`: synthesized audio
- `rtf.csv`: per-sample latency and RTF
- `wer.csv`: per-sample WER plus an average row
- `utmos.csv`: per-file UTMOS scores
- `summary.json`: aggregate metrics and run metadata

## Metric definitions

### WER

- synthesized audio is transcribed with the local Zipformer ASR pipeline
- WER is computed against the reference transcript after normalization

### UTMOSv2

- generated wavs are scored with `utmosv2`
- the aggregate summary stores mean MOS

### RTF

RTF is computed locally during synthesis:

```text
rtf = synthesis_wall_time_seconds / generated_audio_duration_seconds
```

The benchmark stores both per-sample values and aggregate statistics.

## Running WER only

If you already have generated wav files and only want WER:

```bash
uv run python benchmark/inference_wer.py \
  --samples path/to/samples.json \
  --input_audio_dir path/to/generated_wavs \
  --result_csv path/to/wer.csv
```

If you want the older small fixed-sample WER benchmark that also generates audio:

```bash
bash benchmark/run_wer.sh
```

## Running UTMOS only

```bash
bash benchmark/run_utmos.sh
```

## Notes for `linhtran92/viet_bud500`

The dataset card documents:

- built-in `validation` and `test` splits
- `transcription` as the text field
- `audio` as the audio field

Reference: [linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500)

This benchmark only uses the text side of the dataset for generation and scoring.
It does not need the reference audio from `viet_bud500`, and the loader drops the
dataset `audio` column before iteration so it does not depend on `torchcodec`.

## Practical recommendation

Start with `--sample_limit 50` or `--sample_limit 100` to validate the pipeline quickly, then remove the limit for a full validation benchmark.
