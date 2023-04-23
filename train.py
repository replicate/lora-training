import gc
import torch
import os
from lora_diffusion.cli_lora_pti import train as lora_train
from preprocessing import load_and_save_masks_and_captions
from upload import upload_file_to_presigned_url, download_file

from common import (
    random_seed,
    clean_directories,
    extract_zip_and_flatten,
)

COMMON_PARAMETERS = {
    "train_text_encoder": True,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": False,
    "lr_scheduler": "constant",
    "scale_lr": True,
    "lr_warmup_steps": 0,
    "clip_ti_decay": True,
    "color_jitter": True,
    "continue_inversion": False,
    "continue_inversion_lr": 1e-4,
    "initializer_tokens": None,
    "learning_rate_text": 1e-5,
    "learning_rate_ti": 5e-4,
    "learning_rate_unet": 2e-4,
    "lr_scheduler_lora": "constant",
    "lr_warmup_steps_lora": 0,
    "max_train_steps_ti": 700,
    "max_train_steps_tuning": 700,
    "placeholder_token_at_data": None,
    "placeholder_tokens": "<s1>",
    "weight_decay_lora": 0.001,
    "weight_decay_ti": 0,
}


FACE_PARAMETERS = {
    "use_face_segmentation_condition": True,
    "use_template": "object",
    "placeholder_tokens": "<s1>|<s2>",
    "lora_rank": 16,
}

OBJECT_PARAMETERS = {
    "use_face_segmentation_condition": False,
    "use_template": "object",
    "placeholder_tokens": "<s1>|<s2>",
    "lora_rank": 8,
}

STYLE_PARAMETERS = {
    "use_face_segmentation_condition": False,
    "use_template": "style",
    "placeholder_tokens": "<s1>|<s2>",
    "lora_rank": 16,
}

TASK_PARAMETERS = {
    "face": FACE_PARAMETERS,
    "object": OBJECT_PARAMETERS,
    "style": STYLE_PARAMETERS,
}

def main():
    upload_url = os.getenv("upload_url")
    instance_data_url = os.getenv("instance_data_url")
    resolution = os.getenv("resolution", 512)
    task = os.getenv("task", "face")
    train_batch_size = os.getenv("train_batch_size", 1)
    placeholder_tokens = os.getenv("placeholder_tokens", "<s1>")
    max_train_steps = os.getenv("max_train_steps", 700)
    instance_data_folder = os.getenv("instance_data","instance_data")
    output_dir = os.getenv("output_dir", "checkpoints")
    clean_directories([instance_data_folder, output_dir])
    params = {k: v for k, v in TASK_PARAMETERS[task].items()}
    COMMON_PARAMETERS['train_batch_size'] = train_batch_size
    COMMON_PARAMETERS['max_train_steps_ti'] = max_train_steps
    COMMON_PARAMETERS['max_train_steps_tuning'] = max_train_steps
    COMMON_PARAMETERS['placeholder_tokens'] = placeholder_tokens
    
    instance_data=download_file(instance_data_url)
    extract_zip_and_flatten(instance_data, instance_data_folder)
    train_face = task == 'face'
    load_and_save_masks_and_captions(instance_data_folder, instance_data_folder+"/preprocessing",caption_text=placeholder_tokens,target_size=resolution, use_face_detection_instead=train_face)
    using_captions = os.path.isfile("cog_instance_data/preprocessing/caption.txt")
    print(f"Using Captions: {using_captions}")
    COMMON_PARAMETERS['use_mask_captioned_data'] = using_captions
    if using_captions:
        COMMON_PARAMETERS['use_template'] = None
    params.update(COMMON_PARAMETERS)
    seed = random_seed()
    params.update(
        {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "instance_data_dir": instance_data_folder + "/preprocessing",
            "output_dir": output_dir,
            "resolution": resolution,
            "seed": seed,
        }
    )
    lora_train(**params)
    gc.collect()
    torch.cuda.empty_cache()
    
    output_path = output_dir+ "/final_lora.safetensors"
    upload_file_to_presigned_url(output_path, upload_url)

if __name__ == '__main__':
    main()