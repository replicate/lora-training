# Use LoRA PTI Cog

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

```
cog run script/download-weights <your-hugging-face-auth-token>
```

Then, you can run train your dreambooth:

```
cog predict -i instance_data=@quo.zip
```

Resulting file will contain LoRAs that can be used with `patch_pipe` function:

```python
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale, image_grid
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda:1"
)

patch_pipe(pipe, "./step_1000.safetensors")
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

Example Doc on running safetensor PTI outputs at `inference_example.ipynb`
