# Bug Report — Preprocessing Issues

## Issue 1 — `trust_remote_code` no longer supported

### Error
```
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'pnnbao-ump/ngochuyen_voice' isn't
based on a loading script and remove `trust_remote_code`.
```

### Root cause
`datasets` ≥ 2.x dropped support for `trust_remote_code` in `load_dataset()`.
The argument was passed in the original `process_huggingface()` call:
```python
ds = load_dataset(hf_dataset, split=hf_split, trust_remote_code=True)  # broken
```

### Fix applied (`finetune/preprocess.py`)
Removed the argument:
```python
ds = load_dataset(hf_dataset, split=hf_split)
```

---

## Issue 2 — `ImportError: To support decoding audio data, please install 'torchcodec'`

### Error
```
File ".../datasets/features/audio.py", line 186, in decode_example
    raise ImportError("To support decoding audio data, please install 'torchcodec'.")
ImportError: To support decoding audio data, please install 'torchcodec'.
```

### Root cause
Newer versions of `datasets` changed the default audio backend from `torchaudio` to
`torchcodec`. When iterating over a dataset that has an `Audio` feature column, the
library attempts to decode each audio file through `torchcodec`, which is not installed.

### Fix applied (`finetune/preprocess.py`)
Disabled the automatic decode pipeline by casting the audio column with `decode=False`.
This returns the raw `{"bytes": ..., "path": ...}` dict instead of a decoded array, so
we can decode manually with `soundfile` (already a project dependency):

```python
from datasets import Audio, load_dataset
import io, soundfile as sf

ds = ds.cast_column(audio_col, Audio(decode=False))

# Inside the loop:
audio_obj = sample[audio_col]   # {"bytes": bytes | None, "path": str | None}
if audio_obj["bytes"] is not None:
    wav, src_sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32")
else:
    wav, src_sr = sf.read(audio_obj["path"], dtype="float32")
```

This avoids `torchcodec` entirely and works with any `datasets` version.

### Alternative fix (not applied)
Install torchcodec instead:
```bash
uv pip install torchcodec
```
Note: `torchcodec` has strict CUDA/PyTorch version constraints and may conflict with
the existing environment.

---

---

## Issue 3 — `AttributeError: module 'torch.distributed' has no attribute 'tensor'`

### Error
```
File ".../peft/tuners/tuners_utils.py", line 164, in _get_in_out_features
    if _torch_supports_distributed and isinstance(module.weight, torch.distributed.tensor.DTensor):
AttributeError: module 'torch.distributed' has no attribute 'tensor'
```

### Root cause
PEFT 0.18.1 defines:
```python
_torch_supports_dtensor     = version.parse(torch.__version__) >= version.parse("2.5.0")
_torch_supports_distributed = _torch_supports_dtensor and torch.distributed.is_available()
```
With PyTorch 2.5.1 both flags are `True`, so PEFT proceeds to evaluate
`torch.distributed.tensor.DTensor` at line 164.

However, Python only makes a submodule accessible as an attribute of its parent
**after it has been explicitly imported**. `torch.distributed.tensor` is a real
submodule that ships with PyTorch ≥ 2.5, but it is never imported by default when
you do `import torch`. Therefore `torch.distributed.tensor` raises `AttributeError`
even though the module exists on disk.

### Fix applied (`finetune/train.py`)
Pre-import `torch.distributed.tensor` **before** importing `peft`, so the submodule
is registered and the attribute access in PEFT succeeds:

```python
import torch
import torch.distributed.tensor  # pre-import so peft can access torch.distributed.tensor.DTensor
from peft import LoraConfig, TaskType, get_peft_model
```

### Alternative fix (not applied)
Downgrade PEFT to a version that predates the `DTensor` check:
```bash
uv pip install "peft<0.14"
```
Not recommended — older PEFT versions lack bug fixes and features used elsewhere.

---

---

## Issue 4 — `element 0 of tensors does not require grad and does not have a grad_fn`

### Error
```
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
...
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

### Root cause
When `gradient_checkpointing=True`, PyTorch's checkpointing mechanism re-runs the
forward pass during backward and **detaches** the input tensors (embeddings) from
the computation graph to save memory. This means the embedding output no longer has
`requires_grad=True`, so gradients cannot flow back through the LoRA parameters.

This is a known incompatibility between PEFT's LoRA and `gradient_checkpointing`.
Without the workaround, the LoRA adapter weights receive no gradient signal and the
loss cannot be minimised.

### Fix applied (`finetune/train.py`)
Call `model.enable_input_require_grads()` on the base model **before** wrapping it
with `get_peft_model()`. This registers a forward hook on the input embeddings that
sets `requires_grad=True` on their output, keeping them in the graph even after
gradient checkpointing detaches them:

```python
model.enable_input_require_grads()   # ← add this
model = get_peft_model(model, peft_config)
```

This is [documented in the PEFT FAQ](https://huggingface.co/docs/peft/conceptual_guides/troubleshooting#ValueError-Attempting-to-unscale-FP16-gradients) as the required workaround for this combination.

---

## Summary

| # | Problem | Fix |
|---|---------|-----|
| 1 | `trust_remote_code` removed in `datasets` ≥ 2.x | Removed kwarg from `load_dataset()` |
| 2 | `torchcodec` required for audio decode | Cast column with `Audio(decode=False)` + decode via `soundfile` |
| 3 | `torch.distributed.tensor` not accessible as attribute | Pre-import `torch.distributed.tensor` in `train.py` before `peft` |
| 4 | LoRA grads detached by gradient checkpointing | Call `model.enable_input_require_grads()` before `get_peft_model()` |
