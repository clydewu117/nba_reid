#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=UniFormerV2_cam
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=slurm_output/UniFormerV2_cam.out.%j

set -x

CSV_PATH=/fs/scratch/PAS3184/v3/shot_train_test_split_ctrl20.csv
CONFIG_PATH="/users/PAS2985/lei441/nba_reid/outputs/appearance_k400_16frames_motionclassify_ctrl20/config.yaml"
OUTPUT_ROOT=/fs/scratch/PAS3184/v3_cam
MODEL_NAME=UniFormerV2
CHECKPOINTS="/users/PAS2985/lei441/nba_reid/outputs/appearance_k400_16frames_motionclassify_ctrl20/appearance_k400_16frames_motionclassify_ctrl20.pth"

cd /users/PAS2985/lei441/nba_reid-cam
python batch_uniformerv2_cam.py \
  --csv "$CSV_PATH" \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --sampling uniform \
  --modality appearance \
  --methods originalcam \
  --checkpoints "$CHECKPOINTS"
