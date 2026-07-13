#!/usr/bin/env python3
"""SFT LoRA — Qwen3.5-4B-CPT → SFT on Ariadna ml_export data"""
import os, json, torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from huggingface_hub import HfApi

os.environ["TRITON_CACHE_DIR"] = "/workspace/.triton_cache"

CPT_MODEL = "Rob1234567/qwen3.5-4b-ariadna-cpt-v1"
SFT_DATA = "/workspace/sft_data.jsonl"
OUT_MODEL = "Rob1234567/qwen3.5-4b-ariadna-sft-v1"

tok = AutoTokenizer.from_pretrained(CPT_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading CPT model...")
model = AutoModelForCausalLM.from_pretrained(CPT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()

lora_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading SFT data...")
examples = []
for line in open(SFT_DATA):
    obj = json.loads(line)
    msgs = obj.get("messages", [])
    text = tok.apply_chat_template(msgs, tokenize=False)
    examples.append({"text": text})
dataset = Dataset.from_list(examples)
print(f"SFT data: {len(dataset)} examples")

def tokenize_fn(examples):
    result = tok(examples["text"], truncation=True, max_length=2048, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir="/workspace/sft_checkpoints",
    per_device_train_batch_size=8, gradient_accumulation_steps=4,
    num_train_epochs=2, learning_rate=1e-4,
    bf16=True, logging_steps=50, save_steps=250,
    remove_unused_columns=False,
)

print("Starting SFT...")
trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=data_collator)
trainer.train()

model.save_pretrained("/workspace/sft_final")
tok.save_pretrained("/workspace/sft_final")
print("SFT done. Model saved to /workspace/sft_final")

print("Uploading to HF Hub...")
api = HfApi()
api.upload_folder(repo_id=OUT_MODEL, folder_path="/workspace/sft_final")
print(f"Uploaded to {OUT_MODEL}")
