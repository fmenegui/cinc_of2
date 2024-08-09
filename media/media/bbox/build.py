import pandas as pd
from media.bbox.build_yolo_dataset import build_yolo_dataset_from_df
from media.dataset.build_csv import build_csv
from media.dataset.get_labels import add_classes
from media.dataset.validation.stratified import stratified_group_k_fold, add_folds

df = build_csv('/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/ALL/NOISE/more_img')
df = df.sample(frac=1., random_state=42).reset_index(drop=True)
df, unique_labels = add_classes(df, dx_column='dx')
kfold = stratified_group_k_fold(df, ['NORM'], 'header_base', n_splits=5)
df = add_folds(df, kfold)

# build_yolo_dataset_from_df(df, '/home/fdias/repositorios/media/data/yolo_ptbxl_sample', data_perc=1., fold_idx=0)

build_yolo_dataset_from_df(df, '/home/fdias/repositorios/media/data/yolo_ptbxl_full', data_perc=1., fold_idx=0)