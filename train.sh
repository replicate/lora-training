#!/bin/bash

echo "Trigger word: ${TRIGGER_WORD}"
echo "Trigger word: ${CLASS}"
echo "Preprocessing ${PREPROCESSING}"

DATA_URL=$(echo "$DATA_URL" | base64 -d)
UPLOAD_URL=$(echo "$UPLOAD_URL" | base64 -d)
echo "DATA URL: ${DATA_URL}"

echo "Upload URL: ${UPLOAD_URL}"

python3 file_manager.py

if [ "$PREPROCESSING" = "1" ]; then
    export INSTANCE_DIR=${INSTANCE_DIR}/preprocessing
    rm -rf ${INSTANCE_DIR}/caption.txt
fi

echo $INSTANCE_DIR

accelerate config default
#Dreambooth lora
accelerate launch dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of ${TRIGGER_WORD} ${CLASS}" \
  --class_prompt="a photo of ${CLASS}" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=$STEP \
  --validation_prompt="portrait photo of (${TRIGGER_WORD}), ${CLASS}, sharp focus, elegant, render, realistic skin texture, photorealistic, hyper realism, 4k, hdr, smooth" \
  --validation_epochs=100 \
  --seed="0"

python3 evaluate.py

python3 convert-to-safetensors.py --file ${OUTPUT_DIR}/pytorch_lora_weights.bin


