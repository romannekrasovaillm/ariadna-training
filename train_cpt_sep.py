#!/usr/bin/env python3
"""CPT v4: Packing with EOS separators between documents"""
import os, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

os.environ["TRITON_CACHE_DIR"] = "/workspace/.triton_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL = "Qwen/Qwen3.5-4B-Base"
CORPUS = "cpt_train_v1.jsonl"
SEQ_LEN = 2048
MICRO_BS = 1
GRAD_ACCUM = 8
LR = 1e-5
CKPT_EVERY = 1000

if not os.path.exists(CORPUS):
    hf_hub_download('Rob1234567/ariadna-cpt-corpus', CORPUS, repo_type='dataset', local_dir='.')

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
SEP = tok.eos_token_id  # separator between docs

print("Loading & packing corpus (with EOS separators)...")
all_ids = []
with open(CORPUS) as f:
    for line in f:
        try:
            text = json.loads(line)["text"]
            ids = tok.encode(text, truncation=True, max_length=SEQ_LEN-2)
            if len(ids) >= 16:
                all_ids.append(ids)
        except:
            pass

# Pack with EOS separators
packed_seqs = []
total = 0
current = []

for doc_ids in all_ids:
    needed = len(doc_ids) + (1 if current else 0)  # +1 for SEP
    if len(current) + needed <= SEQ_LEN:
        if current:
            current.append(SEP)  # separator before new doc
        current.extend(doc_ids)
    else:
        if len(current) > 0:
            while len(current) < SEQ_LEN:
                current.append(tok.pad_token_id)
            packed_seqs.append(current[:SEQ_LEN])
        current = list(doc_ids)

if len(current) >= 16:
    while len(current) < SEQ_LEN:
        current.append(tok.pad_token_id)
    packed_seqs.append(current[:SEQ_LEN])

total_tokens = sum(1 for seq in packed_seqs for t in seq if t != tok.pad_token_id)
print(f"Corpus: {len(all_ids)} docs → {len(packed_seqs)} packed seqs ({total_tokens/1e6:.1f}M tokens, {total_tokens/(len(packed_seqs)*SEQ_LEN)*100:.0f}% util)")

# Model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
model.gradient_checkpointing_enable()
model.train()
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR)

N = len(packed_seqs)
print(f"CPT Packed+SEP: {N} batches, accum={GRAD_ACCUM}")
opt.zero_grad()
step = 0

for i in range(N):
    seq = packed_seqs[i]
    ids = torch.tensor([seq], device="cuda")
    labels = ids.clone()
    labels[ids == tok.pad_token_id] = -100
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=labels).loss / GRAD_ACCUM
    loss.backward()
    
    if (i+1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
    
    if i % 20 == 0:
        dt = time.time() - t0 if i > 0 else 0.001
        print(f"[{i}/{N} {i/N*100:.1f}%] loss={loss.item()*GRAD_ACCUM:.3f}")
        t0 = time.time()

final_dir = "/workspace/checkpoints/final"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
print(f"✅ Done: {final_dir}")
