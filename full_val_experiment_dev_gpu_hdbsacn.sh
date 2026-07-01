#!/bin/bash
#SBATCH --job-name=reid-hdbscan-dino
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

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p outputs_reid

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Start time: $(date)"
echo "========================================"

# Do NOT use module purge/module load here.
# Your cluster job shell does not have the module command.

source env/bin/activate

export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

CHECKPOINT="./outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt"
PAPER_AIN_CKPT="third_party/Glance-MCMT/deep-person-reid/checkpoints/osnet_ain_ms_m_c.pth.tar"
DATA_ROOT="DataSet/MTMC_Tracking_2025_Preprocessed"

echo "Python:"
which python

echo "Torch:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "Checking DINO checkpoint:"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checking HDBSCAN:"
python - <<'PY'
import hdbscan
print("HDBSCAN import OK")
PY

echo "Import test:"
python -c "from script.con_v1_0.val_experiment import main; print('EXPERIMENT IMPORT OK')"

echo "========================================"
echo "Starting HDBSCAN DINO tracklet diagnostics"
echo "Checkpoint: $CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo "========================================"

for MCS in 3 5 8 10 15 20 30; do
  # OUT_DIR="outputs_reid/hdbscan_dino_mcs${MCS}_$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="outputs_reid/hdbscan_osnet_ain_mcs${MCS}_$(date +%Y%m%d_%H%M%S)"

  echo "----------------------------------------"
  echo "Running HDBSCAN DINO"
  echo "min_cluster_size: $MCS"
  echo "Output dir: $OUT_DIR"
  echo "----------------------------------------"

  python -u -m script.con_v1_0.val_experiment \
    --model_type paper_osnet_ain \
    --paper_model_name osnet_ain_x1_0 \
    --no-paper_pretrained \
    --paper_checkpoint "$PAPER_AIN_CKPT" \
    --data_root "$DATA_ROOT" \
    --split val \
    --out_dir "$OUT_DIR" \
    --level tracklet \
    --tracklet_group_mode auto \
    --aggregation mean_topk \
    --embedding_key bn_embedding \
    --identity_col identity_key \
    --cluster_method hdbscan \
    --min_cluster_size "$MCS" \
    --min_samples 2 \
    --include_occlusion_crops \
    --batch_size 256 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --max_pairs 300000 \
    --pair_sampling balanced \
    --reduce_method pca \
    --make_3d_plots \
    --reduce_3d_method pca \
    --max_plot_points 50000
    # --checkpoint "$CHECKPOINT" \
    # --data_root "$DATA_ROOT" \
    # --split val \
    # --out_dir "$OUT_DIR" \
    # --level tracklet \
    # --tracklet_group_mode auto \
    # --aggregation mean_topk \
    # --embedding_key bn_embedding \
    # --identity_col identity_key \
    # --cluster_method hdbscan \
    # --min_cluster_size "$MCS" \
    # --min_samples 2 \
    # --include_occlusion_crops \
    # --batch_size 256 \
    # --workers "$SLURM_CPUS_PER_TASK" \
    # --max_pairs 300000 \
    # --pair_sampling balanced \
    # --reduce_method pca \
    # --make_3d_plots \
    # --reduce_3d_method pca \
    # --max_plot_points 50000

  echo "Finished MCS=$MCS"
  echo "Metrics: $OUT_DIR/metrics.json"
done

echo "========================================"
echo "Finished all HDBSCAN DINO runs"
echo "Finished time: $(date)"
echo "========================================"