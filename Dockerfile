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

RUN mkdir -p model
RUN mkdir -p pretrained_model
RUN mkdir -p logs
ENV HUGGINGFACE_HUB_CACHE=/challenge/cache
RUN git clone https://huggingface.co/fmenegui/cinc_of1
RUN mv cinc_of1/* model/
RUN git clone https://huggingface.co/fmenegui/cinc_of_pretrained 
RUN mv cinc_of_pretrained/* pretrained_model/

## requirements.txt file.
RUN pip install --upgrade pip
RUN pip install -r media/requirements.txt
RUN pip install -e media
RUN python -c "import timm; model = timm.create_model('convnext_tiny_in22k', pretrained=True)"

