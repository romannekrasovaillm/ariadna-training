# Ariadna Training Pipeline

Полный пайплайн обучения Qwen3.5-4B: курирование 12K arXiv-статей → CPT → SFT → GRPO → GGUF-деплой на RTX 4080.

## Результаты

- **База:** 124K концептов, 14 типов, таксономия v4.0
- **CPT:** 65.8M токенов, loss 6.7→0.85, H100, ~5 ч
- **SFT:** LoRA rank=64, 10K примеров, loss 0.82→0.72, ~2 ч
- **GRPO:** 100 шагов, reward 0.10 (пилот)
- **Модели:** [Rob1234567/qwen3.5-4b-ariadna-cpt-v1](https://huggingface.co/Rob1234567/qwen3.5-4b-ariadna-cpt-v1) → [SFT](https://huggingface.co/Rob1234567/qwen3.5-4b-ariadna-sft-v1) → [GRPO](https://huggingface.co/Rob1234567/qwen3.5-4b-ariadna-grpo-v1)

## Файлы

| Файл | Назначение |
|------|-----------|
| `train_cpt_sep.py` | CPT с пакингом + EOS-разделителями |
| `train_sft_v2.py` | SFT LoRA (merge_and_unload) |
| `rl_direct2.py` | GRPO — model.generate + PPO-clip |
| `chain_launcher.sh` | Автоцепочка CPT→SFT→RL→GGUF |
| `disk_watchdog.sh` | Мониторинг диска на Vast AI |
| `gen_report.py` | Генерация TRAINING_REPORT.md |

## Стек

- **CPT:** PyTorch 2.6 + transformers 4.51 + causal_conv1d + fla 0.4.2
- **SFT:** peft LoRA (rank=64) + DataCollatorForLM
- **RL:** transformers 5.9 + fla 0.5.1 + causal_conv1d 1.6.2 + torch 2.11+cu130

## Воспроизведение

1. Данные: [Rob1234567/ariadna-cpt-corpus](https://huggingface.co/datasets/Rob1234567/ariadna-cpt-corpus)
2. CPT: `python3 train_cpt_sep.py`
3. SFT: `python3 train_sft_v2.py`
4. RL: `python3 rl_direct2.py`

Требуется GPU ≥ 80GB (H100/A100). Для RL — CUDA 13.0 + nvcc.
