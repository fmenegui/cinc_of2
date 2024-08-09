FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

## DO NOT EDIT these 3 lines
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install dependencies 
RUN apt-get update && apt-get install -y git

RUN apt-get update && \
    apt-get install -y git software-properties-common curl
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs
RUN git lfs install

RUN git clone https://huggingface.co/fmenegui/cincX
RUN mv cincX/model .
RUN mv cincX/*.sav model/

## requirements.txt file.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .
RUN python -c "import timm; model = timm.create_model('convnext_tiny_in22k', pretrained=True)"

