#!/bin/bash
#SBATCH -p A5000
#SBATCH --job-name=stance_modernbert_train
#SBATCH --output=logs/stance_modernbert-%j.out
#SBATCH --error=logs/stance_modernbert-%j.err
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

TRAIN_FILE="/home/jzong002/EE6405_Final_Project/data/stance_data_cleaned_train.csv"
TEST_FILE="/home/jzong002/EE6405_Final_Project/data/stance_data_cleaned_test.csv"
OUTPUT_DIR="./output_modernbert"
BATCH_SIZE=16
MAX_LEN=128
N_TRIALS=10
CV_SPLITS=2
GRAD_ACCUM=2
SEED=42
SUBMIT_SLURM=false
INTERACTIVE=false
DRY_RUN=false
SKIP_BASELINE=""
SKIP_TUNING=""
SKIP_FINAL=""
NO_FP16=""

PYTHON_CMD="python ModernBERT_train.py"
PYTHON_CMD="$PYTHON_CMD --train-file $TRAIN_FILE"
PYTHON_CMD="$PYTHON_CMD --test-file $TEST_FILE"
PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --max-len $MAX_LEN"
PYTHON_CMD="$PYTHON_CMD --gradient-accumulation $GRAD_ACCUM"
PYTHON_CMD="$PYTHON_CMD --n-trials $N_TRIALS"
PYTHON_CMD="$PYTHON_CMD --cv-splits $CV_SPLITS"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"
PYTHON_CMD="$PYTHON_CMD $SKIP_BASELINE $SKIP_TUNING $SKIP_FINAL $NO_FP16"

# Run the training script
echo "========================================="
echo "Starting training with command:"
echo "${PYTHON_CMD}"
echo "========================================="
${PYTHON_CMD}

# Print resource usage statistics
echo "========================================="
echo "Resource Usage Statistics:"
scontrol show job ${SLURM_JOB_ID}
echo "========================================="