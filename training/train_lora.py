"""Fine-tune a small instruction model on the RAP//AI JSONL dataset.
Run on a supported GPU/Colab environment; this does not run inside GitHub Pages.
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.getenv("RAP_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.getenv("RAP_DATA_PATH", "training/data/rap_examples.jsonl")
OUT_DIR = os.getenv("RAP_OUTPUT_DIR", "training/output/rap-core-lora")

def main():
    if not torch.cuda.is_available():
        raise SystemExit("Training requires a CUDA GPU. Run this script in a GPU-enabled notebook/runtime.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def format_example(row):
        messages = [
            {"role": "system", "content": "Ты — автор оригинальных русскоязычных рэп-текстов. Пиши выразительно, естественно и без копирования существующих песен."},
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_example)
    config = SFTConfig(
        output_dir=OUT_DIR,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Adapter saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
