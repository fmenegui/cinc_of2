from media.train.load_model import load_model, load_trainer
from media.dataset.base.dataset import BaseDataset, DatasetMeta
from media.dataset.base.dataloader import BaseDataloader
from media.classification.base.classification import BaseClassification
from media.predict.predict_from_dl import predict_and_save
from media.helpers.load_config_module import load_config_module
from media.helpers.evaluate import binary_cm_and_metrics
from media.helpers.compare_model_weights import check_weights
from media.train.prepare_from_config import prepare_df, prepare_resize
from media.helpers.save_timers import save_timers
from media.helpers.log import Tee
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from copy import deepcopy

import time
import pandas as pd
import torch
import lightning as L
import os

def check_variable_in_config(config, attr_name):
    return hasattr(config, attr_name)

def get_dl(config, train_idx, val_idx, test_idx, use_meta=False):
    dl = BaseDataloader(BaseDataset if use_meta is False else DatasetMeta,
        dataframe=config.df,
        file_column=config.image_path_columns,
        label_columns=config.classes,
        batch_size=config.batch_size,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        transforms=config.transforms,
        num_workers=config.num_workers,
        pin_memory=True)
    return dl

def get_train_val_idx(df, fold_idx):
    train_mask = list(df.loc[:,f'fold_{fold_idx}_train'].values.astype(bool))
    valid_mask = list(df.loc[:,f'fold_{fold_idx}_valid'].values.astype(bool))
    test_mask  = list(df.loc[:,'test'].values.astype(bool))
    train_idx = df.index[train_mask].tolist()
    valid_idx = df.index[valid_mask].tolist()
    test_idx  = df.index[test_mask].tolist()
    return train_idx, valid_idx, test_idx


def train_dx_from_config_path(config_path):
    config = load_config_module(config_path)
    train_dx(config)

def train_dx(config):
    '''Log
    '''
    if config.log: logger = Tee(os.path.join(config.save_dir, 'output.log'))
    else: logger = None
    
    config.timers['start'] = time.time()
    print('[INFO] START')
    
    if config.resized_path is None:
        time_tmp = time.time()
        print('[INFO] START Prepare resize')
        config.resized_path = prepare_resize(config)
        config.timers['resize'] = time.time()-time_tmp
        print('[INFO] END Prepare resize\n')
        
    if config.df is None:
        print('[INFO] START Prepare df')
        time_tmp = time.time()
        config.df = prepare_df(config)
        config.timers['prepare_df'] = time.time()-time_tmp
        print('[INFO] END Prepare df\n')
    
    print('[INFO] Save DataFrame')
    time_tmp = time.time()
    config.df.to_csv(os.path.join(config.save_dir, f'df.csv'))
    config.timers['save_df'] = time.time()-time_tmp
    print('[INFO] END Save DataFrame\n')
    
    for i in range(config.kfold_splits):
        if config.folder_index != -1 and config.folder_index != i:
            continue
        
        trainer_kwargs = deepcopy(config.trainer_kwargs)
        
        print('[INFO] START Training fold', i)
        time_tmp = time.time()
        print('-----------------------------------')
        train_indices, val_indices, test_indices = get_train_val_idx(config.df, i)
        
        dl = get_dl(config=config, train_idx=train_indices, val_idx=val_indices, test_idx=test_indices, use_meta=config.use_meta)
        
        model = BaseClassification(
            model=config.model(), 
            loss_fn = config.loss_fn,
            task=config.task,
            optimizer_fn=config.optimizer,
            scheduler_fn = (lambda optimizer: config.scheduler(optimizer, len(train_indices))) if config.scheduler is not None else None,
            metrics_dict=config.metrics_log,
            save_model=config.save_model,
        )
        print(config.save_dir)
        if config.early is not None:
            checkpoint_callback = ModelCheckpoint(dirpath=config.save_dir, save_top_k=1, monitor=config.early['monitor'], mode=config.early['mode'])
            early_stop_callback = EarlyStopping(
                    monitor=config.early['monitor'],  
                    patience=config.early['patience'],  
                    verbose=True,
                    mode=config.early['mode'],  
                    min_delta=config.early['min_delta'] )
        
        if 'callbacks' not in trainer_kwargs.keys(): trainer_kwargs['callbacks'] = []
        if config.early is not None and config.early['use'] is True: trainer_kwargs['callbacks'].append(early_stop_callback)
        if 'best_weights' in config.early and config.early['best_weights'] is True: trainer_kwargs['callbacks'].append(checkpoint_callback)
        
            
        trainer = L.Trainer(logger=True,
                            max_epochs=config.epochs, 
                            default_root_dir=config.save_dir if config.save_dir is not None else os.getcwd(),
                            **trainer_kwargs)
        if config.find_lr:
            tuner = L.pytorch.tuner.Tuner(trainer)
            lr_find_results = tuner.lr_find(model,
                                        train_dataloaders=dl,
                                        min_lr=1e-5,
                                        max_lr=1e0,
                                        early_stop_threshold=None)
            fig = lr_find_results.plot(suggest=True)
            fig.savefig(os.path.join(config.save_dir,'best_lr.png'))

        trainer.fit(model, dl)
        config.timers['train_fold_'+str(i)] = time.time()-time_tmp
        print('[INFO] END Training fold\n', i)
        
        print(f'[INFO] Results (fold {i}): \n', trainer.logged_metrics)
        if 'best_weights' in config.early and config.early['best_weights'] is True:
            model = BaseClassification.load_from_checkpoint(checkpoint_callback.best_model_path, model=config.model(), loss_fn=config.loss_fn)
        else: model = trainer.model
            
        if config.save_model:
            print(f'''[INFO] START Saving model at {os.path.join(config.save_dir, 'model/')}''')
            time_tmp = time.time()
            trainer.save_checkpoint(os.path.join(config.save_dir, f'model/model_fold{i}.ckpt'),
                                    weights_only=True)
            config.timers['save_model_fold_'+str(i)] = time.time()-time_tmp
            print('[INFO] END Saving model\n')
            
            
        # print(f'[INFO] START Predicting Train fold {i}')
        # time_tmp = time.time()
        # predict_and_save(model, dl.train_dataloader(), os.path.join(config.save_dir, f'predictions/train/predictions_train_fold{i}.csv'), task=config.task, classes=config.classes)
        # if config.task=='binary':
        #     assert len(config.classes)==1, 'Binary task must have only one class.'
        #     df_metrics = binary_cm_and_metrics(pd.read_csv(os.path.join(config.save_dir, f'predictions/train/predictions_train_fold{i}.csv')),  
        #                                        true_col=f'Label_{config.classes[0]}', pred_col=f'Prediction_{config.classes[0]}', 
        #                                        threshold=0.5, 
        #                                        save_path=os.path.join(config.save_dir, f'predictions/train/{config.classes[0]}/cm_train_fold{i}.png'))
        # elif config.task=='multilabel':
        #     for _, c in enumerate(config.classes):
        #         df_metrics = binary_cm_and_metrics(pd.read_csv(os.path.join(config.save_dir, f'predictions/train/predictions_train_fold{i}.csv')),  
        #                                 true_col=f'Label_{c}', pred_col=f'Prediction_{c}', 
        #                                 threshold=0.5, 
        #                                 save_path=os.path.join(config.save_dir, f'predictions/train/{c}/cm_train_fold{i}.png'))

        # config.timers['predict_train_fold'+str(i)] = time.time()-time_tmp
        # print(f'[INFO] END Predicting Train fold {i}\n')
        
        
        print(f'[INFO] START Predicting Val fold {i}')
        if check_variable_in_config(config, 'classes_val'): classes_val = config.classes_val
        else: classes_val = config.classes
            
        time_tmp = time.time()
        predict_and_save(model, dl.val_dataloader(), os.path.join(config.save_dir, f'predictions/val/predictions_val_fold{i}.csv'), task=config.task, classes=classes_val)

        if config.task=='binary':
            assert len(classes_val)==1, 'Binary task must have only one class.'
            df_metrics = binary_cm_and_metrics(pd.read_csv(os.path.join(config.save_dir, f'predictions/val/predictions_val_fold{i}.csv')),  
                                               true_col=f'Label_{classes_val[0]}', pred_col=f'Prediction_{classes_val[0]}', 
                                               threshold=0.5, 
                                               save_path=os.path.join(config.save_dir, f'predictions/val/{classes_val[0]}/cm_val_fold{i}.png'))
        elif config.task=='multilabel':
            for _, c in enumerate(classes_val):
                df_metrics = binary_cm_and_metrics(pd.read_csv(os.path.join(config.save_dir, f'predictions/val/predictions_val_fold{i}.csv')),  
                                        true_col=f'Label_{c}', pred_col=f'Prediction_{c}', 
                                        threshold=0.5, 
                                        save_path=os.path.join(config.save_dir, f'predictions/val/{c}/cm_train_fold{i}.png')
                                        )
        config.timers['predict_val_fold'+str(i)] = time.time()-time_tmp
        print(f'[INFO] END Predicting Val fold {i}\n')
        
        
        

        print(df_metrics)
    
    config.timers['global'] = time.time()-config.timers['start']
    save_timers(config.timers, config.save_dir)
    print('[INFO] END\n')
    if logger is not None: logger.close()

# Example usage
if __name__ == "__main__":
    config_path = '/home/fdias/repositorios/media/media/train/config_dx.py'
    train_dx_from_config_path(config_path)
