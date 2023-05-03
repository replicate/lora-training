# LoRA training Cog model

## Use on Replicate

docker build -t dreamboothlora:latest .

docker run --rm --gpus all -it \
  -e MODEL_NAME="runwayml/stable-diffusion-v1-5" \
  -e INSTANCE_DIR="/home/lora-training/instance_data" \
  -e OUTPUT_DIR="/home/lora-training/checkpoints" \
  -e TRIGGER_WORD="telpelight" \
  -e STEP=2500 \
  -e PREPROCESSING=1 \
  -e FACE=1 \
  -e WANDB_API_KEY=<optional> \
  -e DATA_URL="<presigned_url>" \
  -e UPLOAD_URL="<presigned_url>" \
  dreamboothlora:latest 


docker run --rm --gpus all -it \
  -v /home/votiethuy/training-output:/home/lora-training \
  -e MODEL_NAME="runwayml/stable-diffusion-v1-5" \
  -e INSTANCE_DIR="/home/lora-training/instance_data" \
  -e OUTPUT_DIR="/home/lora-training/checkpoints" \
  -e TRIGGER_WORD="banthao" \
  -e CLASS="woman" \
  -e CLASS_DIR="./stable-diffusion-Regularization-Images/sd1.5/woman" \
  -e PREPROCESSING=0 \
  -e FACE=0 \
  -e WANDB_API_KEY=f404879bf0c060ce9b51a5366628019e95748fea \
  -e DATA_URL="<presigned_url>" \
  -e UPLOAD_URL="<presigned_url>" \
  dreambooth:latest