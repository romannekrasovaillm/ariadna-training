#!/bin/bash
# Chain: CPT → SFT → RL → GGUF conversion
cd /workspace

echo "Chain launcher: waiting for CPT to finish..."
while ! ls /workspace/checkpoints/final 2>/dev/null; do sleep 300; done
echo "CPT finished. Starting SFT..."
sleep 30

pip install -q peft trl 2>&1 | tail -2
python3 -u train_sft.py 2>&1 | tee /workspace/sft.log

echo "SFT done. Uploading to HF Hub..."
python3 -c "
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo = 'Rob1234567/qwen3.5-4b-ariadna-sft-v1'
create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path='/workspace/sft_final')
print(f'SFT uploaded to {repo}')
"
echo "SFT_HF_DONE" > /workspace/SFT_HF_DONE

echo "SFT uploaded. Starting RL (GRPO)..."
python3 -u train_grpo.py 2>&1 | tee /workspace/grpo.log

echo "RL done. Uploading GRPO to HF Hub..."
python3 -c "
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo = 'Rob1234567/qwen3.5-4b-ariadna-grpo-v1'
create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path='/workspace/grpo_final')
print(f'GRPO uploaded to {repo}')
"
echo "GRPO_HF_DONE" > /workspace/GRPO_HF_DONE

echo "RL done. Converting to GGUF (ik-llama IQ4_KT)..."
python3 -u convert_to_gguf.py 2>&1 | tee /workspace/convert.log

echo "ALL DONE: CPT→SFT→RL→GGUF complete."
echo "Generating project report..."
python3 -u gen_report.py 2>&1
echo "READY_FOR_DOWNLOAD" > /workspace/DONE
