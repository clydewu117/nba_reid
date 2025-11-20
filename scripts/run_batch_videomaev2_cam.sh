#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=VideoMAEv2_cam
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=slurm_output/VideoMAEv2_cam.out.%j

set -x

CSV_PATH=/fs/scratch/PAS3184/v3/color_train_test_split.csv
CONFIG_PATH=/users/PAS2985/cz2128/ReID/nba_reid/outputs/v3color/VideoMAEv2_mask_beginning_nosplitsampling/config.yaml
OUTPUT_ROOT=/fs/scratch/PAS3184/v3_cam
MODEL_NAME=VideoMAEv2
CHECKPOINTS="/users/PAS2985/cz2128/ReID/nba_reid/outputs/v3color/VideoMAEv2_mask_beginning_nosplitsampling/VideoMAEv2_color_mask_beginning_nosplitsampling.pth"

cd /users/PAS2985/cz2128/ReID/nba_reid-cam
python batch_videomaev2_cam.py \
  --csv "$CSV_PATH" \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --sampling uniform \
  --modality mask \
  --methods originalcam \
  --checkpoints "$CHECKPOINTS"
