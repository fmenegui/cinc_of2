from media.classification.base.classification import BaseClassification
import torch
import xgboost as xgb

def load_model(path):
    return torch.load(path)

def load_trainer(path, model=None): 
    return BaseClassification.load_from_checkpoint(checkpoint_path=path, model=model)

def load_xgboost(path, base_model=None):
    model = xgb.XGBClassifier()  
    model.load_model(path)
    return model