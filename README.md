# LoRA training Cog model

## Use on Replicate

docker build -t dreamboothlora:latest .

docker run --rm --gpus all -it \
  -e MODEL_NAME="runwayml/stable-diffusion-v1-5" \
  -e INSTANCE_DIR="/home/lora-training/instance_data" \
  -e OUTPUT_DIR="/home/lora-training/checkpoints" \
  -e TRIGGER_WORD="<>" \
  -e CLASS_WORD="" \
  -e STEP=2500 \
  -e BATCH_SIZE=1 \
  -e PREPROCESSING=1 \
  -e FACE=1 \
  -e WANDB_API_KEY=<optional> \
  -e DATA_URL="<presigned_url>" \
  -e UPLOAD_URL="<presigned_url>" \
  dreamboothlora:latest 


docker run --rm --gpus all -it \
  -e MODEL_NAME="runwayml/stable-diffusion-v1-5" \
  -e INSTANCE_DIR="/home/lora-training/instance_data" \
  -e OUTPUT_DIR="/home/lora-training/checkpoints" \
  -e TRIGGER_WORD="<>" \
  -e CLASS="woman" \
  -e CLASS_DIR="./stable-diffusion-Regularization-Images/sd1.5/woman" \
  -e PREPROCESSING=0 \
  -e STEP=1500 \
  -e BATCH_SIZE=1 \
  -e FACE=0 \
  -e WANDB_API_KEY=<> \
  -e DATA_URL="<presigned_url>" \
  -e UPLOAD_URL="<presigned_url>" \
  dreambooth:latest


git submodule add https://github.com/kohya-ss/sd-scripts.git