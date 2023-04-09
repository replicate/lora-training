FROM huggingface/transformers-pytorch-gpu

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/huggingface/diffusers.git && cd diffusers && pip3 install .

ADD convert-to-safetensors.py /convert-to-safetensors.py

ADD dreambooth_lora.py /dreambooth_lora.py