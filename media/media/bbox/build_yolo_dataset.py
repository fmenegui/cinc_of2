import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from media.bbox.bbox import read_bbox
from tqdm import tqdm
import math

def get_train_val_idx(df, fold_idx):
    train_mask = list(df.loc[:,f'fold_{fold_idx}_train'].values.astype(bool))
    valid_mask = list(df.loc[:,f'fold_{fold_idx}_valid'].values.astype(bool))
    train_idx = df.index[train_mask].tolist()
    valid_idx = df.index[valid_mask].tolist()
    return train_idx, valid_idx

leads = ['III', 'aVF', 'V3', 'V6', 'II', 'aVL', 'V2', 'V5', 'I', 'aVR', 'V1', 'V4', 'II_long']

def build_yolo_dataset_from_df(df, save_path, data_perc=1., fold_idx=0):
    # Init path
    save_path = os.path.join(save_path, str(fold_idx))
    os.makedirs(save_path, exist_ok=True)
    
    # Save df
    df = df.sample(frac=data_perc, random_state=42).reset_index(drop=True)
    df.to_csv(os.path.join(save_path, 'df.csv'), index=False)
    
    # Create Train/Valid folders
    for folder in ['train', 'valid']: os.makedirs(os.path.join(save_path, folder), exist_ok=True)
    save_path_train = os.path.join(save_path, 'train')
    save_path_valid = os.path.join(save_path, 'valid')
    
    # Get Train/Valid Idx
    train_idx, valid_idx = get_train_val_idx(df, fold_idx)
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_valid = df.loc[valid_idx].reset_index(drop=True)
    
    # Gen dataset
    # dataset_train = build_dataset_coco_format(df_train, save_path_train)
    # save_coco_dataset(dataset_train, os.path.join(save_path_train, '_annotations.coco.json'))    
    
    dataset_valid = build_dataset_coco_format(df_valid, save_path_valid)
    save_coco_dataset(dataset_valid, os.path.join(save_path_valid, '_annotations.coco.json'))
    
    pass


def build_dataset_coco_format(df, save_path):
    dataset =      {
                    "images": [],
                    "annotations": [],
                    }
    categories = [{'id': idx+1, 
                  'name': f'{lead}', 
                   'supercategory': 'Lead'} for idx, lead in enumerate(leads)]
    dataset['categories'] = categories
    
    for idx in tqdm(range(0, len(df))):
        # if idx < 849: continue
        # image name
        image_name = df['image_name'].loc[idx]

        # data
        data = _read(df, idx)
        image = data['image']
        image.save(os.path.join(save_path, image_name))
        image_height, image_width = image.height, image.width
        
        # add image
        dataset['images'].append({
        "id": idx + 1, 
        "width": image_width,
        "height": image_height,
        "file_name": image_name
        })
        
        # add bbox
        bbox = data['bbox']
        for lead in leads:
            if lead in bbox.keys():
                xmin, ymin, xmax, ymax = bbox[lead]['lead']
         
                xmin, ymin, xmax, ymax =  _correct_bbox_coords([xmin, ymin, xmax, ymax], image_width, image_height)
                if xmin is None: continue
                x, y, w, h = _convert_bbox_to_coco_format(xmin, ymin, xmax, ymax)
                area = _calculate_area(w, h)
                bbox_coco = [x, y, w, h]
                annotation = {
                    "id": len(dataset['annotations']) + 1,
                    "image_id": idx+1,
                    "category_id": leads.index(lead) + 1,
                    "segmentation": [],
                    "area": area,
                    "bbox": bbox_coco,
                    "iscrowd": 0,
                }
                dataset['annotations'].append(annotation)
    return dataset
    pass

def save_coco_dataset(dataset, save_path):
    with open(save_path, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)


def _correct_bbox_coords(bbox, width, height):
    def mpl_to_pil_coords(x_mpl, y_mpl):
        x_pil = x_mpl
        y_pil = height - y_mpl  
        return int(x_pil), int(y_pil)

    if any(x == float('inf') or x == float('-inf') or math.isnan(x) for x in bbox) : return None, None, None, None
    x_min, y_min, x_max, y_max = bbox

    x_min, y_min = mpl_to_pil_coords(x_min, y_min)
    x_max, y_max = mpl_to_pil_coords(x_max, y_max)

    x_min, x_max = sorted([x_min, x_max])
    y_min, y_max = sorted([y_min, y_max])

    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, width), min(y_max, height)

    return (x_min, y_min, x_max, y_max)

def _convert_bbox_to_coco_format(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]

def _calculate_area(width, height):
    return width * height

def _read(df, idx):
    image_name = df['image_name'].loc[idx]
    img_path = df['image_full_path'].loc[idx]
    base_img = df['image_full_path'].loc[idx].split(image_name)[0]

    return {'image':Image.open(img_path).convert("RGB"),
            'bbox':read_bbox(image_name.split('.png')[0], base_img)
            }
def _mpl_to_pil_coords(x_mpl, y_mpl, fig_height):
    x_pil = x_mpl
    y_pil = fig_height - y_mpl  
    return int(x_pil), int(y_pil)

            
