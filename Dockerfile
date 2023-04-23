FROM huggingface/transformers-pytorch-gpu

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/huggingface/diffusers.git && cd diffusers && pip3 install .

RUN git clone https://github.com/cloneofsimo/lora.git && cd lora && pip3 install .

ADD convert-to-safetensors.py /convert-to-safetensors.py

ADD upload.py /upload.py

ADD preprocessing.py /preprocessing.py

ADD dreambooth_lora.py /dreambooth_lora.py

ADD train.sh /train.sh

ADD train.py train.py

# CMD [ "./train.sh" ]