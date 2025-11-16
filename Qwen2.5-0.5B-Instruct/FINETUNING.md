# Fine-tuning Documentation

**Prerequisites**
- Python 3.10+ recommended. Ensure CUDA drivers + toolkit are installed on GPU server if using GPU.
- From the `Qwen2.5-0.5B-Instruct` directory, install dependencies:

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
  --model_name_or_path qwen-2.5-0.5b-instruct \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5
```

**Saving Output**
- The `--output` directory:
  - `pytorch_model.bin` / model weights (or `adapter` files for PEFT)
  - `tokenizer` files
  - `config.json`
  - `label_mappings.json`
- Save training args and commit hash alongside results:

```bash
# after run, from project dir
cp git.commit results/qwen_model/
pip freeze > results/qwen_model/requirements.freeze.txt
```