from media.predict.predict_from_dl import predict_and_save
from media.train.load_model import load_trainer
from media.helpers.load_config_module import load_config_module
from media.dataset.base.dataset import BaseDataset
from media.dataset.base.dataloader import BaseDataloader
from media.helpers.evaluate import binary_cm_and_metrics
import os
import glob
import pandas as pd

def load_dx_model(model_folder, verbose):
    filenames = glob.glob(os.path.join(model_folder, 'model*.*ckpt'))
    models = [load_trainer(filename, config.model()).model for filename in filenames]
    return models

def get_dl(config, df, train_idx, val_idx, test_idx):
    dl = BaseDataloader(BaseDataset,
        dataframe=df,
        file_column=config.image_path_columns,
        label_columns=config.classes,
        batch_size=config.batch_size,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        transforms=config.transforms,
        pin_memory=True)
    return dl


'''Configs'''
config_path_dx = 'config_dx.py'
config = load_config_module(config_path_dx)
config.image_path_columns = ['path']
config.group_column = 'ID'
config.classes = ['Normal']


'''Dataset Dra. Natália'''
df = pd.read_csv('/home/fdias/data/dataset_ecg_dranatalia/anotacao_natalia_final20k.csv')
resized_path = '/home/fdias/data/dataset_ecg_dranatalia_resized224/resized/'
df['path'] = df['path'].apply(lambda x: os.path.join(resized_path, x))
df = df.sample(frac=1., random_state=42).reset_index(drop=True)

'''Load models'''
# models = load_dx_model('/home/fdias/repositorios/media/experiments/train_submissao/logs/Normal/(early)2024-04-01_18-27-30/model', False)
models = load_dx_model('/home/fdias/repositorios/media/experiments/train_submissao/logs/Normal/1stSubmission2024-03-21_09-40-52/model', False) # first submission

'''Predict'''
for i, model in enumerate(models):
    print(f'model {i}')
    dl = get_dl(config, df, df.index.tolist(), df.index.tolist(), df.index.tolist())
    predict_and_save(model, dl.test_dataloader(), f'predictions_external_dranatalia/predictions_fold{i}.csv', task=config.task)

dfs = []
for i in range(len(models)): dfs.append(pd.read_csv(f'predictions_external_dranatalia/predictions_fold{i}.csv'))
mean_predictions = pd.concat([df['Predictions'] for df in dfs], axis=1).mean(axis=1)
mean_labels = pd.concat([df['Labels'] for df in dfs], axis=1).mean(axis=1)
mean_df = pd.DataFrame({'Predictions': mean_predictions, 'Labels': mean_labels})
    
df_metrics = binary_cm_and_metrics(mean_df,  
                                               true_col='Labels', pred_col='Predictions', 
                                               threshold=0.5, 
                                               save_path=f'predictions_external_dranatalia/cm.png')
df_metrics.to_csv('predictions_external_dranatalia/df.csv')
print(df_metrics)