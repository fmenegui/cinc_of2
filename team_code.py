#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from media.helpers.load_config_module import load_config_module
from media.train.train_dx import train_dx
import glob
from PIL import Image
import torch
# from helper_code import load_image

from helper_code import *

config_path_dx = 'config_dx.py'
config = load_config_module(config_path_dx)

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    train_dx_model(data_folder, model_folder, verbose)
    train_digitization_model(data_folder, model_folder, verbose)
        
def train_digitization_model(data_folder, model_folder, verbose):
    
    from helper_code import find_records
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()
        
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    # Train the model.
    if verbose:
        print('Training the model on the data...')
    config.data_img_dir = data_folder
    config.resized_path = None
    config.save_dir = model_folder
    config.save_model = True
    config.data_perc = 1.
        
    train_dx(config)

    if verbose:
        print('Done.')
        print()
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_trainer(filename, model):
    def adjust_key(k): return k[len('model.'):]
    print(filename)
    checkpoint = torch.load(filename)
    state_dict = checkpoint['state_dict']
    adjusted_state_dict = {adjust_key(k): v for k, v in state_dict.items()}
    model.load_state_dict(adjusted_state_dict, strict=False) 
    return model


def load_models(model_folder, verbose):
    return load_digitization_model(model_folder, verbose), load_dx_model(model_folder, verbose)

def load_digitization_model(model_folder, verbose):
    import joblib
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

def load_dx_model(model_folder, verbose):
    # filename = os.path.join(model_folder, 'model/model_fold0.ckpt')
    filenames = list(glob.glob(os.path.join(model_folder, 'model*.*ckpt'))) + list(glob.glob(os.path.join(model_folder, 'model/*.ckpt')))
    models = [load_trainer(filename, config.model()) for filename in filenames]
    return models

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    labels = run_dx_model(classification_model, record, None, verbose)
    signal = run_digitization_model(digitization_model, record, verbose)
    
    return signal, labels

def run_dx_model(dx_model, record, signal, verbose):
    models = dx_model
    classes = config.classes
    
    images = load_images(record)
    labels = []
    for img in images:
        pred_prob_list = []
        for model in models:
            img = images[0]
            img = img.convert('RGB')
            img = img.resize((config.size, config.size), Image.Resampling.LANCZOS)
            img = config.transforms['test'](image=np.array(img))['image']
            
            img = img.unsqueeze(0)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            img = img.to(device)
            
            with torch.no_grad(): 
                pred = model(img)
            pred_probs = torch.sigmoid(pred)
            pred_prob_list.append(pred_probs)
        # print(pred_prob_list)
        tensors_cpu = [t.cpu() for t in pred_prob_list]
        concatenated = torch.cat(tensors_cpu)
        average = torch.mean(concatenated, dim=0)
        preds = average > config.threshold
        # print(preds)
        # print('\n')
        labels = [c for c, p in zip(classes, preds) if bool(p) is True]
        # print(labels)

    return labels

def run_digitization_model(digitization_model, record, verbose):
    from helper_code import get_header_file, load_text, get_num_samples, get_num_signals
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1, high=1, size=(num_samples, num_signals))

    return signal

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    save_digitization_model(model_folder, digitization_model)
    save_dx_model(model_folder, classification_model, classes)
        
def save_digitization_model(model_folder, model):
    import joblib
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    return None