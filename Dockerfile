FROM huggingface/transformers-pytorch-gpu

RUN apt-get update -y && apt-get install -y vim

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/huggingface/diffusers.git && cd diffusers && pip3 install .

RUN git clone https://github.com/cloneofsimo/lora.git && cd lora && pip3 install .

ADD convert-to-safetensors.py /convert-to-safetensors.py

ADD file_manager.py /file_manager.py

ADD preprocessing.py /preprocessing.py

ADD dreambooth_lora.py /dreambooth_lora.py

ADD train.sh /train.sh

ADD train.py /train.py

ADD evaluate.py /evaluate.py

CMD [ "./train.sh" ]