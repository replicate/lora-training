import os
import gc
import mimetypes
import shutil
from zipfile import ZipFile
import torch

from cog import BasePredictor, Input, Path
from lora_diffusion.cli_lora_pti import train as lora_train
from lora_diffusion import (
    UNET_DEFAULT_TARGET_REPLACE,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
)


COMMON_PARAMETERS = {
    "train_text_encoder": True,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": False,
    "lr_scheduler": "constant",
    "scale_lr": True,
    "lr_warmup_steps": 0,
    "use_8bit_adam": False,
    "clip_ti_decay": True,
    "color_jitter": True,
    "continue_inversion": False,
    "continue_inversion_lr": 1e-4,
    "device": "cuda",
    "initializer_tokens": None,
    "learning_rate_text": 1e-5,
    "learning_rate_ti": 5e-4,
    "learning_rate_unet": 1e-4,
    "lora_rank": 4,
    "lr_scheduler_lora": "constant",
    "lr_warmup_steps_lora": 0,
    "max_train_steps_ti": 500,
    "max_train_steps_tuning": 1000,
    "perform_inversion": True,
    "placeholder_token_at_data": None,
    "placeholder_tokens": "<s1>|<s2>",
    "weight_decay_lora": 0.001,
    "weight_decay_ti": 0,
}


FACE_PARAMETERS = {
    "use_face_segmentation_condition": True,
    "use_template": "object",
}

OBJECT_PARAMETERS = {
    "use_face_segmentation_condition": False,
    "use_template": "object",
}

STYLE_PARAMETERS = {
    "use_face_segmentation_condition": False,
    "use_template": "style",
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
        ),
        task: str = Input(
            choices=["face", "object", "style"],
            description="Type of LoRA model you want to train",
        ),
        seed: int = Input(description="A seed for reproducible training", default=None),
        resolution: int = Input(
            description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=512,
        ),
    ) -> Path:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        params = {k: v for k, v in TASK_PARAMETERS[task].items()}
        params.update(COMMON_PARAMETERS)
        params["resolution"] = resolution
        params["seed"] = seed

        cog_instance_data = "cog_instance_data"
        cog_output_dir = "checkpoints"
        for path in [cog_instance_data, cog_output_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # extract zip contents, flattening any paths present within it
        with ZipFile(str(instance_data), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, cog_instance_data)

        lora_train(**params)
        gc.collect()
        torch.cuda.empty_cache()

        num_steps = COMMON_PARAMETERS["max_train_steps_tuning"]
        return Path(cog_output_dir) / f"step_{num_steps}.safetensors"
