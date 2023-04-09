# LoRA training Cog model

## Use on Replicate

Easy-to-use model pre-configured for faces, objects, and styles:

[![Replicate](https://replicate.com/replicate/lora-training/badge)](https://replicate.com/replicate/lora-training)

Advanced model with all the parameters:

[![Replicate](https://replicate.com/replicate/lora-advanced-training/badge)](https://replicate.com/replicate/lora-advanced-training)

Feed the trained model into this inference model to run predictions:

[![Replicate](https://replicate.com/replicate/lora/badge)](https://replicate.com/replicate/lora)

If you want to share your trained LoRAs, please join the `#lora` channel in the [Replicate Discord](https://discord.gg/replicate).

## Use locally

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

```
cog run script/download-weights <your-hugging-face-auth-token>
```

Then, you can run train your dreambooth:

```
cog predict -i instance_data=@my-images.zip
```

The resulting LoRA weights file can be used with `patch_pipe` function:

```python
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale, image_grid
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda:1"
)

patch_pipe(pipe, "./my-images.safetensors")
prompt = "detailed photo of <s1><s2>, detailed face, a brown cloak, brown steampunk corset, belt, virtual youtuber, cowboy shot, feathers in hair, feather hair ornament, white shirt, brown gloves, shooting arrows"

tune_lora_scale(pipe.unet, 0.8)
tune_lora_scale(pipe.text_encoder, 0.8)

imgs = pipe(
    [prompt],
    num_inference_steps=50,
    guidance_scale=4.5,
    height=640,
    width=512,
).images
...
```

docker pull huggingface/transformers-pytorch-gpu

docker run --rm --gpus all -it \
   -v /home/votiethuy/diffusers:/home/diffusers \
   -v /home/votiethuy/lora-training:/home/lora-training \
   huggingface/transformers-pytorch-gpu


python3 /home/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /home/lora-training/checkpoints/final_lora.safetensors  --dump_path save_dir --from_safetensors

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/lora-training/train_data/ainn_training"
export OUTPUT_DIR="/home/lora-training/checkpoints/ainn"

#Dreambooth lora
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of ain nguyen girl" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=700 \
  --validation_prompt="A photo of ain nguyen girl" \
  --validation_epochs=50 \
  --seed="0"

04/08/2023 05:39:47 - INFO - __main__ - Distributed environment: NO
04/08/2023 05:52:06

Convert to WEBUI format
python3 convert-to-safetensors.py --file pytorch_lora_weights.bin