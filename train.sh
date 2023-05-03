#!/bin/bash

echo "Trigger word: ${TRIGGER_WORD}"
echo "Trigger word: ${CLASS}"
echo "Preprocessing ${PREPROCESSING}"

python3 file_manager.py

if [ "$PREPROCESSING" = "1" ]; then
    export INSTANCE_DIR=${INSTANCE_DIR}/preprocessing
    rm -rf ${INSTANCE_DIR}/caption.txt
fi

echo $INSTANCE_DIR

accelerate config default
#Dreambooth lora
# accelerate launch dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="${TRIGGER_WORD}" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=500 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --report_to="wandb" \
#   --max_train_steps=$STEP \
#   --validation_prompt="portrait photo of (${TRIGGER_WORD}), sharp focus, elegant, render, realistic skin texture, photorealistic, hyper realism, 4k, hdr, smooth" \
#   --validation_epochs=100 \
#   --seed="0"

accelerate launch dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of ${TRIGGER_WORD}" \
  --class_prompt="a photo of ${CLASS}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompt="portrait photo of (${TRIGGER_WORD}), ${CLASS},sharp focus, elegant, render, realistic skin texture, photorealistic, hyper realism, 4k, hdr, smooth" \
  --validation_epochs=100 \
  --num_class_images=200 \
  --max_train_steps=800

# python3 evaluate.py

# python3 convert-to-safetensors.py --file ${OUTPUT_DIR}/pytorch_lora_weights.bin


