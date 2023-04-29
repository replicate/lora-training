#!/bin/bash

echo "Trigger word: ${TRIGGER_WORD}"
echo "Preprocessing ${PREPROCESSING}"

python3 file_manager.py

if [ "$PREPROCESSING" = "1" ]; then
    export INSTANCE_DIR=$INSTANCE_DIR/preprocessing
    rm -rf $INSTANCE_DIR/preprocessing/caption.txt
fi

echo $INSTANCE_DIR

accelerate config default
#Dreambooth lora
accelerate launch dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${TRIGGER_WORD}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=500 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=$STEP \
  --validation_prompt="portrait photo of (${TRIGGER_WORD}), sharp focus, elegant, render, realistic skin texture, photorealistic, hyper realism, 4k, hdr, smooth" \
  --validation_epochs=100 \
  --seed="0"

python3 evaluate.py