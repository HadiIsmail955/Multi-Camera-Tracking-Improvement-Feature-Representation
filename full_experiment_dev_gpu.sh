#!/bin/bash
#SBATCH --job-name=reid-train
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd $SLURM_SUBMIT_DIR

mkdir -p logs
mkdir -p outputs_reid

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Start time: $(date)"
echo "========================================"

module purge
module load devel/python/3.11.7-gnu-11.4

source env/bin/activate

export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "Python:"
which python

echo "Torch:"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "Import test:"
python -c "from script.con_v1_0.train_experiment import main; print('EXPERIMENT IMPORT OK')"

echo "Starting training..."
python -u -m script.con_v1_0.train_experiment

echo "========================================"
echo "Finished time: $(date)"
echo "========================================"