from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
from file_manager import upload_file_to_presigned_url
from safetensors.torch import save_file
from report import send_training_report
def test_and_upload():
    trigger_word = os.getenv("TRIGGER_WORD")
    output_dir = os.getenv("OUTPUT_DIR")
    upload_url = os.getenv("UPLOAD_URL")
    modelid = os.getenv("MODEL_NAME")
    file_path = output_dir+"/pytorch_lora_weights.bin"
    pt_state_dict = torch.load(file_path)
    file_safetensors = output_dir+"/{}.safetensors".format(trigger_word)
    save_file(pt_state_dict, file_safetensors, metadata={"format": "pt"})
    pipe = DiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.load_lora_weights(output_dir)
    
    prompt = "portrait photo of {},  winter forest, natural skin texture, 4k textures, soft cinematic light, adobe lightroom, intricate, elegant, highly detailed, sharp focus, cinematic look, soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, hdr, faded".format(trigger_word)
    negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=512,
                    height=768, num_inference_steps=25).images[0]
    img_path = output_dir + "/out-1.png"
    image.save(img_path)
    upload_file_to_presigned_url(file_safetensors,upload_url)
    data = {"status": "success", "message": "Train {} completed".format(os.environ.get("WANDB_NAME")), "error_code": ""}
    send_training_report(data)

if __name__ == '__main__':
    try:
        test_and_upload()
    except Exception as error:
        print(error)
        data = {"status": "fail", "message": f"{error}", "error_code": "unexpected_error"}
        send_training_report(data)