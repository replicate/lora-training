#!/bin/bash
echo "VAST Instance: ${VAST_CONTAINERLABEL}"
echo "Trigger word: ${TRIGGER_WORD}"
echo "Trigger word: ${CLASS}"
echo "Preprocessing ${PREPROCESSING}"
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

accelerate config default
#Dreambooth lora
accelerate launch dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of ${TRIGGER_WORD} ${CLASS}" \
  --num_class_images=200 \
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

if [ -n "$VAST_CONTAINERLABEL" ]; then
  # Do something if the variable is set
  echo "VAST_CONTAINERLABEL is set to: $VAST_CONTAINERLABEL"
  cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; 
  ./vast destroy instance ${VAST_CONTAINERLABEL:2} 
fi