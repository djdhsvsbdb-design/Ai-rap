# RAP//AI model training

This folder contains a small **starter** dataset and a LoRA fine-tuning script.

## Current status

- The JSONL file contains a few original seed examples for testing the pipeline.
- The script targets `Qwen/Qwen2.5-0.5B-Instruct`.
- No model has been trained yet. A successful run requires a CUDA-enabled GPU runtime.
- The browser app still uses the local procedural `RAP Core`; it does not load this adapter yet.

## Run in a GPU notebook/runtime

From the repository root:

```bash
pip install -r training/requirements.txt
python training/train_lora.py
```

The output adapter is written to `training/output/rap-core-lora`. Keep generated model weights out of Git unless there is a deliberate reason to publish them; they can be large.

## Improve the dataset

Add one JSON object per line to `data/rap_examples.jsonl` with:
- `instruction`: a clear writing request (topic, style, form, mood, line count)
- `output`: an original answer written for that request

Use only text you wrote or have permission to use. Do not add copyrighted song lyrics or imitate a specific living artist. Keep a separate validation set to compare outputs before and after fine-tuning.

## Method

The script uses supervised fine-tuning (SFT) with a LoRA adapter via Hugging Face TRL/PEFT. LoRA trains a small set of additional parameters while leaving the base model weights frozen. Training quality depends heavily on dataset size, variety, and quality.
