# LoRA training Cog model

## Use on Replicate

Easy-to-use model pre-configured for faces, objects, and styles:

[![Replicate](https://replicate.com/replicate/lora-training/badge)](https://replicate.com/replicate/lora-training)

Advanced model with all the parameters:

[![Replicate](https://replicate.com/replicate/lora-advanced-training/badge)](https://replicate.com/replicate/lora-advanced-training)

Feed the trained model into this inference model to run predictions:

[![Replicate](https://replicate.com/replicate/lora/badge)](https://replicate.com/replicate/lora)

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
