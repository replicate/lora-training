# LoRA training Cog model

## Use on Replicate

gcloud batch jobs submit lora-train-job-gpu \
  --location us-central1 \
  --config container-job-gpu.json