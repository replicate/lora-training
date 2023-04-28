#!/bin/bash

python3 file_manager.py

rm -rf $INSTANCE_DIR/preprocessing/caption.txt

accelerate config default
#Dreambooth lora
accelerate launch dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR/preprocessing \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="ainn" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=500 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$STEP \
  --validation_prompt="ainn" \
  --validation_epochs=100 \
  --seed="0"

python3 evaluate.py