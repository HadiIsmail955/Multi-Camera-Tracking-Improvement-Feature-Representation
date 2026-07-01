#!/bin/bash
#SBATCH --job-name=reid-tracklet-diagnosis
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
mkdir -p outputs_full_reid_diagnosis_tracklet

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

CHECKPOINT="./outputs_reid/dinov2_reid_embedding_v2_20260613_211015/checkpoints/last.pt"
DATA_ROOT="DataSet/MTMC_Tracking_2025_Preprocessed"
# OUT_DIR="outputs_reid/outputs_full_reid_diagnosis_tracklet_paper/$(date +%Y%m%d_%H%M%S)"
PAPER_CKPT="third_party/Glance-MCMT/deep-person-reid/checkpoints/osnet_ain_ms_m_c.pth.tar"
PAPER_AIN_CKPT="third_party/Glance-MCMT/deep-person-reid/checkpoints/osnet_ain_ms_m_c.pth.tar"
OUT_DIR="outputs_reid/final_paper_text_osnet_ain_val_tracklet_$(date +%Y%m%d_%H%M%S)"

echo "Python:"
which python

echo "Torch:"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "Checking checkpoint:"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available files in outputs_reid:"
    ls -lah outputs_reid || true
    exit 1
fi

echo "Import test:"
python -c "from script.con_v1_0.val_experiment import main; print('EXPERIMENT IMPORT OK')"

echo "========================================"
echo "Starting tracklet-level ReID diagnostics..."
echo "Checkpoint: $CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUT_DIR"
echo "========================================"

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
  --cluster_method dbscan \
  --dbscan_eps 0.025 \
  --min_samples 2 \
  --include_occlusion_crops \
  --batch_size 256 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --max_pairs 300000 \
  --pair_sampling balanced \
  --reduce_method pca \
  --make_3d_plots \
  --reduce_3d_method pca \
  --max_plot_points 500000

  # --checkpoint "$CHECKPOINT" \
  # --data_root "$DATA_ROOT" \
  # --split val \
  # --out_dir "$OUT_DIR" \
  # --level tracklet \
  # --tracklet_group_mode auto \
  # --aggregation mean_topk \
  # --embedding_key bn_embedding \
  # --identity_col identity_key \
  # --cluster_method dbscan \
  # --min_samples 2 \
  # --run_eps_grid \
  # --dbscan_eps 0.0475 \
  # --eps_values 0.0425 0.045 0.0475 0.05 \
  # --include_occlusion_crops \
  # --batch_size 256 \
  # --workers "$SLURM_CPUS_PER_TASK" \
  # --max_pairs 300000 \
  # --pair_sampling balanced \
  # --reduce_method pca \
  # --make_3d_plots \
  # --reduce_3d_method pca \
  # --max_plot_points 50000

  # --scenes "Warehouse_015"

  # --checkpoint "$CHECKPOINT" \
  # --data_root "$DATA_ROOT" \
  # --split val \
  # --scenes "$SCENE" \
  # --out_dir "$OUT_DIR" \
  # --level crop \
  # --embedding_key bn_embedding \
  # --identity_col identity_key \
  # --cluster_method dbscan \
  # --dbscan_eps 0.35 \
  # --min_samples 2 \
  # --run_eps_grid \
  # --include_occlusion_crops \
  # --max_eval_samples 10000 \
  # --eval_sampling identity_balanced \
  # --batch_size 256 \
  # --workers $SLURM_CPUS_PER_TASK \
  # --max_pairs 50000 \
  # --pair_sampling balanced \
  # --reduce_method pca \
  # --reduce_3d_method pca \
  # --max_plot_points 3000 \

echo "========================================"
echo "Finished time: $(date)"
echo "Outputs saved in: $OUT_DIR"
echo "Important files:"
echo "  $OUT_DIR/metrics.json"
echo "  $OUT_DIR/dbscan_eps_grid.csv"
echo "  $OUT_DIR/misclustered_points.csv"
echo "  $OUT_DIR/merge_errors.csv"
echo "  $OUT_DIR/fragmentation_errors.csv"
echo "  $OUT_DIR/interactive_miscluster_diagnosis.html"
echo "========================================"