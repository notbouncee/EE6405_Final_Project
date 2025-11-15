#!/bin/bash
#SBATCH -p A5000
#SBATCH --job-name=stance_bert_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=22G

# Print job information
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================="

# Create necessary directories
mkdir -p logs
mkdir -p output
mkdir -p data

module load Miniforge3  
conda activate stanceDetect  

# Set up environment variables
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${HOME}/.cache/torch"
export TOKENIZERS_PARALLELISM=false 

# Print GPU information
echo "========================================="
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "========================================="


# Print package versions
echo "========================================="
echo "Package Versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')"
echo "========================================="

# Data file paths - CHANGE THESE TO YOUR FILE PATHS
TRAIN_FILE="/home/jzong002/EE6405_Final_Project/data/reddit_posts_and_comments_train.csv"
TEST_FILE="/home/jzong002/EE6405_Final_Project/data/reddit_posts_and_comments_test.csv"

# Output directory for results
OUTPUT_DIR="output/reddit"

# Model configuration
MODEL_NAME="bert-base-uncased"  # Options: bert-base-uncased, roberta-base, albert-base-v2, distilbert-base-uncased
MAX_LEN=512                      # Maximum sequence length

# Training configuration
BATCH_SIZE=16                    # Training batch size
EPOCHS=3                         # Number of training epochs

# Hyperparameter optimization
N_TRIALS=20                      # Number of Optuna trials (increase for better optimization)
CV_SPLITS=3                      # Number of cross-validation splits

CMD="python BERT_train.py"
CMD="${CMD} --train-file ${TRAIN_FILE}"
CMD="${CMD} --test-file ${TEST_FILE}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --model-name ${MODEL_NAME}"
CMD="${CMD} --max-len ${MAX_LEN}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --n-trials ${N_TRIALS}"
CMD="${CMD} --cv-splits ${CV_SPLITS}"

# Run the training script
echo "========================================="
echo "Starting training with command:"
echo "${CMD}"
echo "========================================="
${CMD}

# Print resource usage statistics
echo "========================================="
echo "Resource Usage Statistics:"
scontrol show job ${SLURM_JOB_ID}
echo "========================================="
