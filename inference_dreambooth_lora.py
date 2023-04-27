from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
from upload import upload_file_to_presigned_url
from safetensors.torch import save_file

def test_and_upload():
    trigger_word = os.getenv("TRIGGER_WORD")
    output_dir = os.getenv("OUTPUT_DIR")
    upload_url = os.getenv("UPLOAD_URL")
    file_path = output_dir+"/pytorch_lora_weights.bin"
    pt_state_dict = torch.load(file_path)
    file_safetensors = output_dir+"/pytorch_lora_weights.safetensors"
    save_file(pt_state_dict, file_safetensors, metadata={"format": "pt"})
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.load_attn_procs(output_dir)
    
    prompt = "A picture of {}".format(trigger_word)
    image = pipe(prompt, num_inference_steps=25).images[0]
    img_path = "/tmp/out-1.png"
    image.save(img_path)
    upload_file_to_presigned_url(file_safetensors,upload_url)

if __name__ == '__main__':
    test_and_upload()