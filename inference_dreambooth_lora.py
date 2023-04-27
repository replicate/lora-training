from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os

def inference():
    trigger_word = os.getenv("TRIGGER_WORD")
    output_dir = os.getenv("OUTPUT_DIR")
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.unet.load_attn_procs(output_dir)
    
    prompt = "A picture of {}".format(trigger_word)
    image = pipe(prompt, num_inference_steps=25).images[0]

if __name__ == '__main__':
    inference()