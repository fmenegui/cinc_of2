'''Imports
'''
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torchmetrics
import os
from media.helpers.copy_file import copy_script_to_new_folder
from media.helpers.create_dir import create_timestamped_directory
import pandas as pd
from media.classification.callbacks import BatchAccumulatedMetricsCallback

'''Configs'''

'''Save
'''
name = 'finetune_fromInCorEMG/BCE/384/10folds/BALANCE_DE_CLASSES'
pretrained_model = 'convnext_tiny_in22k' 
save_dir = 'logs'
save_model = True
save_script = False
log = False
timers = {}


'''Dataset
'''
data_img_dir = '/mnt/experiments1/felipe.dias/CINC_CHALLENGE_2024/OFICIAL/images'
resized_path = '/mnt/experiments1/felipe.dias/CINC_CHALLENGE_2024/OFICIAL/images384'
size = 384
data_perc = 1.
num_workers = 2

'''Prediction
'''
threshold = 0.5

''' Training config
'''
find_lr = True
image_path_columns = ['image_full_path']
group_column = 'header_base'
classes = 'NORM,Acute MI,Old MI,STTC,CD,HYP,PAC,PVC,AFIB/AFL,TACHY,BRADY'.split(',')
task = 'multilabel' if len(classes) > 1 else 'binary'
f1_score_avg = 'macro' if len(classes) > 1 else 'binary'
print(f'Task: {task}\nF1_score_avg: {f1_score_avg}')
epochs = 30
batch_size = 32
lr = 1e-4
kfold_splits = 10
folder_index = 0 # -1: all, 1,2,3...: folder index

# early=None
early = {
    'use': True,
    'monitor':'val_f1',
    'verbose':True,
    'patience':6,
    'mode':'max',
    'min_delta':0.001,
    'best_weights':False
}

# Include it in your Trainer's callbacks
from sklearn.metrics import f1_score
trainer_kwargs = {
                  'precision':'16-mixed',
                  'callbacks':[BatchAccumulatedMetricsCallback(metric_to_function_dict={"f1": (f1_score, {'average': 'macro'})})],
                  'gradient_clip_val':1.0,
                  'gradient_clip_algorithm':'norm'
                  }

''' Model
'''
loss_fn = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([2.29125499, 104.30143541,   4.10527307,   4.16408787,
         4.45059208,   8.22914307,  54.77135678,  19.07174103,
        13.88471338,  25.14302191,  34.22135008])) 
optimizer = lambda parameters: torch.optim.AdamW(parameters, lr=lr)
scheduler = None

name = os.path.join(name, pretrained_model)
ckp_path = 'pretrained_model/pretrained_model_fold0.ckpt'
def model():
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict']
    model = timm.create_model('convnext_tiny_in22k', pretrained=False, num_classes=13)
    model.load_state_dict({k[len('model.'):]: v for k, v in state_dict.items()})
    in_features = model.get_classifier().in_features
    model.reset_classifier(len(classes))
    model.classifier = torch.nn.Linear(in_features, len(classes))
    return model
use_meta = False


''' Evaluation
'''
metrics_log = None
metrics_nolog = None


''' Augmentations
'''
try:
    cfg = model.default_cfg
    mean, std = cfg['mean'], cfg['std']
except:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

t_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.1),
        A.ToGray(),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(16, 16), always_apply=True),
        A.Normalize(mean=mean, std=std),  
        ToTensorV2()
    ])
t = A.Compose(
        [
            A.ToGray(),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(16, 16), always_apply=True),
            A.Normalize(mean=mean, std=std),  
            ToTensorV2(),
        ]
)
transforms = {
        "train": t_train,
        "valid": t,
        "test": t,
    }


''' CSV
'''
df = None

'''Save script
'''
# save_dir = create_timestamped_directory(save_dir, name)
# if save_script:  copy_script_to_new_folder(__file__, save_dir)
