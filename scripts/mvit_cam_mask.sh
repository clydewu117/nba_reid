CSV_PATH=/fs/scratch/PAS3184/v3/train_test_split.csv
OUTPUT_ROOT=/fs/scratch/PAS3184/baicheng_cam/new_w
MODEL_NAME=MViTv2

python batch_mvitv2_cam.py \
  --csv "$CSV_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --sampling uniform \
  --modality mask \
  --methods originalcam \
  --checkpoints \
    /fs/scratch/PAS3184/baicheng_ckpt/new2/mask_k400_16frames_beginning_nosplitsampling_drop02/best_model.pth