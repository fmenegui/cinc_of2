# media

> Repositório para códigos da competição "George B. Moody PhysioNet Challenge 2024: Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024"

## Start here

### Criação de ambiente geração de imagem (`myenv`)
`conda create -n myenv python=3.10` \
`conda activate myenv` \
`conda install git` \
`pip install -r requirements.txt` \
`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz` \


### Criação de ambiente media (`media`)
`conda create -n media python=3.11` \ (3.10.13 -> docker)
`conda activate media` \
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` \
`pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110` \
`conda install git`
`pip install -e .`

### Criação de ambiente media2 (`media2`)
`conda create -n media2 python=3.10` \ (3.10.13 -> docker)
`conda activate media2` \
`conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` \
`conda install git`
`pip install -e .`

### Run Docker
`docker build -t your_image_name .`
`docker run --gpus all -it your_image_name bash`

### Test submission data dir:
prepare_submission.sh
`/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION`
## Challenge

### Link do Challenge
> https://moody-challenge.physionet.org/2024/