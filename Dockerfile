FROM huggingface/transformers-pytorch-gpu:4.29.1

RUN apt-get update -y && apt-get install -y vim wget

RUN wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; 

ADD requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/huggingface/diffusers.git && cd diffusers && pip3 install .

RUN git clone https://github.com/cloneofsimo/lora.git && cd lora && pip3 install .

ADD convert-to-safetensors.py /app/convert-to-safetensors.py

ADD file_manager.py /app/file_manager.py

ADD preprocessing.py /app/preprocessing.py

ADD dreambooth_lora.py /app/dreambooth_lora.py

ADD dreambooth.py /app/dreambooth.py

ADD train.sh /app/train.sh

ADD train.py /app/train.py

ADD evaluate.py /app/evaluate.py

ADD report.py /app/report.py

WORKDIR /app

CMD [ "./train.sh" ]