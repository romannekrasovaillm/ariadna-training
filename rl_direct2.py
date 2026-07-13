import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

print("Loading...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

policy = AutoModelForCausalLM.from_pretrained("/workspace/sft_final", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", ignore_mismatched_sizes=True)
ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B-Base", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
ref.eval()
for p in ref.parameters(): p.requires_grad = False

prompts, golds = [], []
for line in open("/workspace/classify_sample.jsonl"):
    obj = json.loads(line)
    prompts.append(obj["prompt"])
    golds.append(obj["meta"]["type"])

K, lr, beta, steps = 4, 1e-6, 0.01, min(100, len(prompts))
opt = torch.optim.AdamW(policy.parameters(), lr=lr)
policy.train()

print(f"GRPO: {steps} steps")
for step in range(steps):
    prompt, gold = prompts[step], golds[step]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(policy.device)
    
    with torch.no_grad():
        outputs = policy.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.9, num_return_sequences=K, pad_token_id=tok.pad_token_id)
    completions = [tok.decode(o, skip_special_tokens=True) for o in outputs]
    rewards = torch.tensor([1.0 if gold.lower() in c.lower() else 0.1 for c in completions])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    for i in range(K):
        ids = tok(prompt + completions[i], return_tensors="pt", truncation=True, max_length=1024).input_ids.to(policy.device)
        
        with torch.no_grad():
            ref_out = ref(ids.cpu()).logits
            ref_logp = F.log_softmax(ref_out, -1)
            ref_token_lp = ref_logp[:, :-1].gather(-1, ids.cpu()[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        pol_out = policy(ids).logits
        pol_logp = F.log_softmax(pol_out, -1)
        pol_token_lp = pol_logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        ratio = torch.exp(pol_token_lp - ref_token_lp.to(policy.device))
        clipped = torch.clamp(ratio, 0.8, 1.2)
        kl = (pol_token_lp - ref_token_lp.to(policy.device)).pow(2)
        loss = -torch.min(ratio * adv[i], clipped * adv[i]).mean() + beta * kl.mean()
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    
    if step % 10 == 0:
        print(f"[{step}] loss={loss.item():.4f} reward={rewards.mean():.2f}")

policy.save_pretrained("/workspace/grpo_final")
tok.save_pretrained("/workspace/grpo_final")
print("✅ GRPO done")
