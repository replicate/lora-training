#!/bin/bash
echo "VAST Instance: ${VAST_CONTAINERLABEL}"
echo "Trigger word: ${TRIGGER_WORD}"
echo "Preprocessing ${PREPROCESSING}"
echo "REPORT_URL ${REPORT_URL}"
echo "REPORT_TOKEN ${REPORT_TOKEN}"
echo "WANDB_NAME ${WANDB_NAME}"
BATCH_SIZE="${BATCH_SIZE:=1}"
echo "Batch Size: ${BATCH_SIZE}"
RESOLUTION="${RESOLUTION:=512}"
echo "RESOLUTION: ${RESOLUTION}"

padding_needed=$(( (4 - ${#DATA_URL} % 4) % 4 ))
encoded_string_with_padding="${DATA_URL}$(printf '%*s' $padding_needed | tr ' ' '=')"
DATA_URL=$(echo "$encoded_string_with_padding" | base64 -d)
echo "DATA URL: ${DATA_URL}"

padding_needed=$(( (4 - ${#UPLOAD_URL} % 4) % 4 ))
encoded_string_with_padding="${UPLOAD_URL}$(printf '%*s' $padding_needed | tr ' ' '=')"
UPLOAD_URL=$(echo "$encoded_string_with_padding" | base64 -d)
echo "Upload URL: ${UPLOAD_URL}"

python3 file_manager.py

if [ "$PREPROCESSING" = "1" ]; then
    export INSTANCE_DIR=${INSTANCE_DIR}/preprocessing
    rm -rf ${INSTANCE_DIR}/caption.txt
fi

echo $INSTANCE_DIR

accelerate config default --config_file /app/accelerate.yaml
#Dreambooth lora
accelerate launch --mixed_precision="fp16" --zero_stage=3 dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="a photo of ${TRIGGER_WORD}" \
  --output_dir=$OUTPUT_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=1 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=$STEP \
  --validation_prompt="portrait photo of (${TRIGGER_WORD}), sharp focus, elegant, render, realistic skin texture, photorealistic, hyper realism, 4k, hdr, smooth" \
  --validation_negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation" \
  --validation_epochs=100 \
  --seed="0"

python3 evaluate.py

# python3 convert-to-safetensors.py --file ${OUTPUT_DIR}/pytorch_lora_weights.bin

if [ -n "$VAST_CONTAINERLABEL" ]; then
  # Do something if the variable is set
  echo "VAST_CONTAINERLABEL is set to: $VAST_CONTAINERLABEL"
  cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; 
  ./vast destroy instance ${VAST_CONTAINERLABEL:2} 
fi