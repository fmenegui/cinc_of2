
# Docker
-> `bash prepare_submission.sh [diretório do experimento]`
-> copiar modelo de digitalizacao para pasta model
-> trocar save_dir=False em config_dx.py
-> trocar classes=['Normal'] em config_dx.py
-> `docker build -t image_dummy .`

`docker run --shm-size 50G --gpus all -it --network none -v /home/fdias/CHALLENGE_SUBMISSIONS/github/tmp3/cinc2/model:/challenge/model -v /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/test4/00000:/challenge/test_data -v /home/fdias/CHALLENGE_SUBMISSIONS/github/tmp3/cinc2/test_outputs -v /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/train/00000:/challenge/training_data image_dummy bash`

`docker run --shm-size 50G --gpus all -it --network none -v /home/fdias/CHALLENGE_SUBMISSIONS/github/tmp4/cinc2/model:/challenge/model -v /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/test4/00000:/challenge/test_data -v /home/fdias/CHALLENGE_SUBMISSIONS/github/tmp4/cinc2/test_outputs -v /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/train/00000:/challenge/training_data image_dummy bash`

python train_model.py -d training_data -m model

python run_model.py -d training_data -m model -o test_outputs

python evaluate_model.py -d training_data -o test_outputs

# Local
python train_model.py -d /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/train -m model

python run_model.py -d /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/test4 -m model -o test_outputs

python evaluate_model.py -d /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/test4 -o test_outputs


# Local 2
python train_model.py -d /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/train/00000 -m model2

python run_model.py -d /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/test4 -m model2 -o test_outputs
