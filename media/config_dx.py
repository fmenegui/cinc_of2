'''Imports
'''
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import albumentations as A
import timm
import torchmetrics

from media.helpers.copy_file import copy_script_to_new_folder
from media.helpers.create_dir import create_timestamped_directory
from lightning.pytorch.callbacks import ModelCheckpoint
from media.dataset.validation.stratified import stratified_group_k_fold, add_folds
import time
'''Configs'''

'''Save
'''
name = 'Normal'
save_dir = '/home/fdias/repositorios/media/experiments/train_submissao/logs'
save_model = True
save_script = True
timers = {}


'''Dataset
'''
size = 224
data_img_dir = '/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/ALL/NOISE/more_img'
resized_path = '/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/ALL/NOISE_224/more_img'
data_perc = 1.

'''Prediction
'''
threshold = 0.5

''' Training config
'''
image_path_columns = ['image_full_path']
group_column = 'header_base'
classes = ['NORM']
epochs = 30
batch_size = 64
lr = 1e-4
kfold_splits = 5
folder_index = -1 # -1: all, 1,2,3...: folder index
# checkpoint_callback = ModelCheckpoint(
#     dirpath=os.path.join(save_dir,'checkpoints/'), # Directory where checkpoints are saved
#     filename='{epoch}-{val_loss:.2f}', # Checkpoint file name structure
#     monitor='val_loss', # Metric to monitor for improvement
#     save_top_k=3, # Number of best checkpoints to save (based on metric)
#     mode='min', # Mode for metric monitoring ('min' for minimum, 'max' for maximum)
#     every_n_epochs=1, # Save checkpoint every N epochs
# )
# trainer_kwargs = {'callbacks': [checkpoint_callback]}
trainer_kwargs = {'precision':16}
trainer_kwargs = {}

''' Model
'''
loss_fn = None
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.3])) # nn.CrossEntropyLoss()
optimizer = lambda parameters: torch.optim.AdamW(parameters, lr=lr)
scheduler = None 
# scheduler = lambda optimizer, len_loader: torch.optim.lr_scheduler.OneCycleLR(optimizer, 
#                                                    max_lr=lr, 
#                                                    total_steps=epochs*len_loader)
model = lambda: timm.create_model('convnext_tiny_in22k', pretrained=True, num_classes=len(classes))
# model = lambda: timm.create_model('convnext_large_in22k', pretrained=True, num_classes=len(classes))
# model = lambda: timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True, num_classes=len(classes))
# model = lambda: timm.create_model('beit_large_patch16_512.in22k_ft_in22k_in1k', pretrained=True, num_classes=len(classes))

# model = lambda: timm.create_model('resnet18', pretrained=True, num_classes=len(classes)) # baseline

''' Evaluation
'''
task = 'binary'
metrics_log = {
    'acc': torchmetrics.Accuracy(task=task, num_classes=len(classes)),
    'f1': torchmetrics.F1Score(task=task, num_classes=len(classes))           
}
metrics_nolog = {
    'cm': torchmetrics.ConfusionMatrix(task=task, num_classes=len(classes))
}


''' Augmentations
'''
try:
    cfg = model.default_cfg
    mean, std = cfg['mean'], cfg['std']
except:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
t = A.Compose(
        [
            # A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=mean, std=std),  
            ToTensorV2(),
        ]
)
transforms = {
        "train": t,
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