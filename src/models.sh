#!/bin/bash
#SBATCH --job-name=sm_ob2_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300000
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/lustre/users/johngachoka/SOIL-MOISTURE-PREDICTION/outputs/logs/slurm_%x_%j.out
#SBATCH --error=/scratch/lustre/users/johngachoka/SOIL-MOISTURE-PREDICTION/outputs/logs/slurm_%x_%j.err

# --- Usage ---
# sbatch models.sh src/configs/rf_baseline.yaml
# sbatch models.sh src/configs/xgb_tuned.yaml --stride 2

# --- Working Directory Setup ---
PROJECT_DIR=/scratch/lustre/users/johngachoka/SOIL-MOISTURE-PREDICTION
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

CONFIG_PATH="${1:?Error: Please provide a config file. Usage: sbatch models.sh src/configs/rf_baseline.yaml}"
shift || true
EXTRA_ARGS="$@"

# --- Environment Setup ---
module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate /scratch/lustre/users/$USER/envs/mkfe_env

# --- Diagnostics ---
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Running from: $(pwd)"
echo "Using python: $(which python)"
echo "Config: $CONFIG_PATH"
echo "Extra args: $EXTRA_ARGS"

# --- Output Directory Creation ---
mkdir -p outputs/logs
mkdir -p outputs/figures
mkdir -p outputs/metrics
mkdir -p models
mkdir -p data/predictions

# --- Run Objective 2 ML Pipeline ---
python -u -m src.train --config "$CONFIG_PATH" $EXTRA_ARGS

echo "Job finished on: $(date)"