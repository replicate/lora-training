import gc
import torch
import os
from cog import BasePredictor, Input, Path
from lora_diffusion.cli_lora_pti import train as lora_train
from upload import upload_file_to_presigned_url, download_file, url_local_fn

from common import (
    random_seed,
    clean_directories,
    extract_zip_and_flatten,
    get_output_filename,
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

class Predictor(BasePredictor):
    def predict(
        self,
        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
            default=None
        ),
        instance_data_url: str = Input(description="Instance Data URL for training", default=None),
        task: str = Input(
            default="face",
            choices=["face", "object", "style"],
            description="Type of LoRA model you want to train",
        ),
        upload_url: str = Input(description="Upload URL for upload model", default=None),
        seed: int = Input(description="A seed for reproducible training", default=None),
        train_batch_size: int = Input(description="train batch size", default=1),
        max_train_steps: int = Input(description="max train steps ti", default=700),
        placeholder_tokens: str = Input(
            default="<s1>|<s2>",
            description="The placeholder tokens to use for the initializer. If not provided, will use the first tokens of the data.",
        ),
        resolution: int = Input(
            description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=512,
        ),
    ) -> Path:
        if seed is None:
            seed = random_seed()
        print(f"Using seed: {seed}")

        cog_instance_data = "cog_instance_data"
        cog_output_dir = "checkpoints"
        clean_directories([cog_instance_data, cog_output_dir])

        params = {k: v for k, v in TASK_PARAMETERS[task].items()}
        COMMON_PARAMETERS['train_batch_size'] = train_batch_size
        COMMON_PARAMETERS['max_train_steps_ti'] = max_train_steps
        COMMON_PARAMETERS['max_train_steps_tuning'] = max_train_steps
        COMMON_PARAMETERS['placeholder_tokens'] = placeholder_tokens
        

        if instance_data is None:
            instance_data=download_file(instance_data_url)
        extract_zip_and_flatten(instance_data, cog_instance_data)
        using_captions = os.path.isfile("cog_instance_data/caption.txt")
        print(f"Using Captions: {using_captions}")
        COMMON_PARAMETERS['use_mask_captioned_data'] = using_captions
        if using_captions:
            COMMON_PARAMETERS['use_template'] = None
        params.update(COMMON_PARAMETERS)
        params.update(
            {
                "pretrained_model_name_or_path": "./stablediffusion15",
                "instance_data_dir": cog_instance_data,
                "output_dir": cog_output_dir,
                "resolution": resolution,
                "seed": seed,
            }
        )
        lora_train(**params)
        gc.collect()
        torch.cuda.empty_cache()

        num_steps = COMMON_PARAMETERS["max_train_steps_tuning"]
        
        weights_path = Path(cog_output_dir) / f"step_{num_steps}.safetensors"
        output_path = Path(cog_output_dir) / get_output_filename(instance_data)
        weights_path.rename(output_path)
        upload_file_to_presigned_url(output_path, upload_url)
        return output_path
