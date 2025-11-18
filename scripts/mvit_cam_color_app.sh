CSV_PATH=/fs/scratch/PAS3184/v3/color_train_test_split.csv
OUTPUT_ROOT=/fs/scratch/PAS3184/baicheng_cam/new_w
MODEL_NAME=MViTv2

python batch_mvitv2_cam.py \
  --csv "$CSV_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --model-name "$MODEL_NAME" \
  --sampling uniform \
  --modality appearance \
  --methods originalcam \
  --checkpoints \
    /fs/scratch/PAS3184/baicheng_ckpt/new2/color_appearance_k400_16frames_beginning_nosplitsampling_drop03/best_model.pth