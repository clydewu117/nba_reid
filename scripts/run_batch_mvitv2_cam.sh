#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=MViTv2_cam
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=slurm_output/MViTv2_cam.out.%j

set -x

CSV_PATH=/fs/scratch/PAS3184/v3/shot_train_test_split_ctrl20.csv
CONFIG_PATH=/fs/scratch/PAS3184/eccv_mvitv2_cam/appearance_drop02_ctrl20_good/config.yaml
OUTPUT_ROOT=/fs/scratch/PAS3184/v3_cam
MODEL_NAME=MViTv2
CHECKPOINTS="/fs/scratch/PAS3184/eccv_mvitv2_cam/appearance_drop02_ctrl20_good/appearance_drop02_ctrl20_good.pth"

cd /users/PAS2099/clydewu117/nba_reid-cam
python batch_mvitv2_cam.py \
  --csv "$CSV_PATH" \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --sampling uniform \
  --modality appearance \
  --methods originalcam gradcam \
  --checkpoints "$CHECKPOINTS"
