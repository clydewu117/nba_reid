#!/bin/bash
#SBATCH --account=pas2136
#SBATCH --job-name=MViTv2_cam
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.12122@osu.edu
#SBATCH --output=slurm_output/MViTv2_cam.out.%j

set -x

CSV_PATH=/fs/scratch/PAS3184/v3/train_test_split.csv
OUTPUT_ROOT=/fs/scratch/PAS3184/v3_cam_mvit
MODEL_NAME=MViTv2
CHECKPOINTS="\
  /path/to/mvit_checkpoint1.pth \
  /path/to/mvit_checkpoint2.pth \
"

cd /users/PAS2985/cz2128/ReID/nba_reid-cam
python batch_mvitv2_cam.py \
  --csv "$CSV_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --methods originalcam scorecam \
  --checkpoints $CHECKPOINTS

