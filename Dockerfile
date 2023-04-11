FROM huggingface/transformers-pytorch-gpu

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/huggingface/diffusers.git && cd diffusers && pip3 install .

RUN git clone https://github.com/votiethuy/autocrop.git && cd autocrop && pip3 install .

RUN pip3 install git+https://github.com/elliottzheng/face-detection.git@master

ADD convert-to-safetensors.py /convert-to-safetensors.py

ADD preprocessing.py /preprocessing.py

ADD dreambooth_lora.py /dreambooth_lora.py

ADD train.sh /train.sh

CMD [ "./train.sh" ]