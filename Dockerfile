FROM huggingface/transformers-pytorch-gpu

RUN apt-get update -y && apt-get install -y vim wget

RUN git clone https://github.com/tobecwb/stable-diffusion-Regularization-Images.git

RUN wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; 

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/huggingface/diffusers.git && cd diffusers && pip3 install .

RUN git clone https://github.com/cloneofsimo/lora.git && cd lora && pip3 install .

ADD convert-to-safetensors.py /convert-to-safetensors.py

ADD file_manager.py /file_manager.py

ADD preprocessing.py /preprocessing.py

ADD dreambooth_lora.py /dreambooth_lora.py

ADD dreambooth.py /dreambooth.py

ADD train.sh /train.sh

ADD train.py /train.py

ADD evaluate.py /evaluate.py

ADD report.py /report.py

CMD [ "./train.sh" ]