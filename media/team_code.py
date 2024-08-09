#!/usr/bin/env python

import numpy as np
import os
from media.helpers.load_config_module import load_config_module
from media.train.train_dx import train_dx
from media.train.load_model import load_trainer
from helper_code import load_image
import torch
from PIL import Image
import glob

config_path_dx = 'config_dx.py'
config = load_config_module(config_path_dx)

# Train your digitization model.
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

# Train your dx classification model.
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

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    import joblib
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    # filename = os.path.join(model_folder, 'model/model_fold0.ckpt')
    filenames = glob.glob(os.path.join(model_folder, 'model*.*ckpt'))
    models = [load_trainer(filename, config.model()).model for filename in filenames]
    return models
    # return load_trainer(filename, config.model()).model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
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
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    models = dx_model
    classes = config.classes
    
    images = load_image(record)
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
        preds = torch.Tensor(pred_prob_list).mean(dim=0) > config.threshold
        label = 'Normal' if preds == True else 'Abnormal'
        labels.append(label)

    return labels


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def extract_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    import joblib
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    return None
    
    
if __name__ == '__main__':
    train_dx_model('/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/TEST_SUBMISSION/train/', 
                   'model', 
                   True)
    
    model = load_dx_model('/home/fdias/repositorios/media/logs/first_submission_test/2024-03-16_14-37-23', 
                          True)
    
    run_dx_model(model, 
                    '/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/ALL/CLEAN/MORE_IMG/records100/00000/00169_lr.hea',
                    None, 
                    True)