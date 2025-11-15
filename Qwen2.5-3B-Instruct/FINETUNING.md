# Fine-tuning Documentation

This document describes the fine-tuning process used in this project (keep current fine-tuning).
It captures environment setup, reproducible commands, logging, and tips for large-model finetuning.

**Purpose**
- Fine-tune a pretrained Transformer (Qwen2.5/3B-style checkpoint) for stance detection on Reddit/stanced datasets.
- Provide reproducible commands and tips so training can be repeated or audited.

**Prerequisites**
- Python 3.10+ recommended. Ensure CUDA drivers + toolkit are installed on GPU server if using GPU.
- From the `Qwen2.5-3B-Instruct` directory, install dependencies:

```bash
# inside server user home/project dir
cd ~/EE6405_Final_Project/EE6405_Final_Project/Qwen2.5-3B-Instruct
pip install --user -r requirements.txt
# optional libs for memory-efficient finetuning
pip install --user accelerate peft bitsandbytes
```

**Record environment (recommended)**
```bash
# save python packages and git state for reproducibility
pip freeze > requirements.freeze.txt
git rev-parse --short HEAD > git.commit
```

**Typical training command (full fine-tuning)**
Replace paths with server paths and set `--model_name_or_path` if you want a specific pretrained checkpoint.

```bash
python train_qwen_stance.py \
  --csv /usr1/home/s125mdg21_03/EE6405_Final_Project/EE6405_Final_Project/data/preprocessed/reddit_posts_and_comments_train.csv \
  --output ./results/qwen_model \
  --model_name_or_path qwen-2.5-3b-instruct \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5
```

**Recommended flags for GPU and stability**
- Use FP16 to reduce memory: `--fp16` (script must pass to `TrainingArguments`) or set `training_args.fp16=True`.
- Use `gradient_accumulation_steps` when effective batch size needed.
- If memory is tight, reduce `--batch-size` and increase accumulation steps.

**PEFT / LoRA (optional, recommended for large models)**
- PEFT/LoRA updates a small number of parameters and drastically reduces memory and compute needs.
- Install: `pip install --user peft bitsandbytes accelerate`
- Example (after adding support in the training script):

```bash
python train_qwen_stance.py \
  --csv /path/to/train.csv \
  --output ./results/qwen_model_lora \
  --model_name_or_path qwen-2.5-3b-instruct \
  --use_lora \
  --lora_r 8 \
  --epochs 3 \
  --batch-size 16 \
  --fp16
```

**Saving artifacts**
- The `--output` directory should contain:
  - `pytorch_model.bin` / model weights (or `adapter` files for PEFT)
  - `tokenizer` files
  - `config.json`
  - `label_mappings.json` (if the script writes it). If missing, use `extract_label_mapping.py`.
- Save training args and commit hash alongside results:

```bash
# after run, from project dir
cp git.commit results/qwen_model/
pip freeze > results/qwen_model/requirements.freeze.txt
```

**Logging & experiment tracking**
- Script may support W&B (`wandb`) — set `WANDB_PROJECT` and login before running.
- Alternatively, redirect stdout/stderr to a log file:

```bash
python train_qwen_stance.py --csv ... --output ./results/qwen_model > results/qwen_model/train.log 2>&1
```

**Resuming / Recovering label mappings**
- If a checkpoint lacks `label_mappings.json`, run:

```bash
python extract_label_mapping.py /path/to/checkpoint --output /path/to/checkpoint/label_mappings.json
```

**Quick checklist for reproducible runs**
- [ ] Note GPU model and memory
- [ ] Save `git` commit and `pip freeze`
- [ ] Save exact command used (put in `results/<run>/command.txt`)
- [ ] Save `label_mappings.json` with model outputs

**Notes & tips**
- Fine-tuning (the current script's default) is appropriate — training from scratch is not recommended for these model sizes.
- If you want help adding explicit `--save_training_args` or automatic environment capture to `train_qwen_stance.py`, I can patch the script to write `training_args.json` into the output directory.

---
Document created to make fine-tuning steps reproducible. If you'd like, I can also:
- Add a `run_train.sh` template to the folder
- Patch `train_qwen_stance.py` to automatically save args and environment info
- Add PEFT/LoRA example integrated into the script

Tell me which of the above you want next.