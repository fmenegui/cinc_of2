"""
Transforms dataset from COCO format to YOLO v8 format.

- The script restructures a dataset from the COCO annotation format to the YOLO v8 annotation format. 
- It processes the 'train', 'valid', and optionally 'test' directories within a specified COCO dataset folder. 
Each directory should contain image files and a COCO format JSON annotation file named '_annotations.coco.json'. 
The script generates a corresponding YOLO v8 structured dataset with separate directories for images and labels (annotations)
for each of the dataset splits ('train', 'valid', 'test'). 
A 'data.yaml' file is also generated to describe the dataset configuration for YOLO v8 training.

COCO file structure expected:
    main_folder:
        train/
            _annotations.coco.json
            *.jpg/png
        valid/
            _annotations.coco.json
            *.jpg/png
        test/ (optional)
            _annotations.coco.json
            *.jpg/png

YOLO v8 file structure generated:
    main_folder:
        data.yaml
        train/
            images/
                *.jpg/png
            labels/
                *.txt
        valid/
            images/
                *.jpg/png
            labels/
                *.txt
        test/
            images/
                *.jpg/png
            labels/
                *.txt (optional, if annotations are available)

The generated 'data.yaml' includes paths to the image directories for each dataset split, the number of classes ('nc'), and the list of class names ('names').


Example usage:
    coco_main_folder = '/path/to/coco/main_folder'
    yolo_main_folder = '/path/to/yolo/main_folder'
    convert_coco_to_yolo_v8(coco_main_folder, yolo_main_folder)

"""
import json
import os
import shutil
import sys
import yaml
from tqdm import tqdm

def process_annotations(json_file, labels_folder, categories, cat_map):
    with open(json_file) as f:
        annotations = json.load(f)

    for anno in tqdm(annotations['annotations'], desc=f"Processing {os.path.basename(json_file)}"):
        image_info = next((img for img in annotations['images'] if img['id'] == anno['image_id']), None)
        if not image_info:
            continue
        category_idx = cat_map[anno['category_id']]
        bbox = anno['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / image_info['width']
        y_center = (bbox[1] + bbox[3] / 2) / image_info['height']
        width = bbox[2] / image_info['width']
        height = bbox[3] / image_info['height']
        yolo_annotation = f"{category_idx} {x_center} {y_center} {width} {height}"
        annotation_path = os.path.join(labels_folder, os.path.splitext(image_info['file_name'])[0] + '.txt')
        with open(annotation_path, 'a') as f:
            f.write(yolo_annotation + '\n')

def copy_images(src_folder, images_folder):
    image_files = list(filter(lambda f: f.endswith(('.jpg', '.jpeg', '.png')), os.listdir(src_folder)))
    for img_file in tqdm(image_files, desc=f"Copying images from {os.path.basename(src_folder)}"):
        src_path = os.path.join(src_folder, img_file)
        dst_path = os.path.join(images_folder, img_file)
        shutil.copy2(src_path, dst_path)

def process_set(coco_main_folder, yolo_main_folder, part, categories, cat_map):
    coco_part_folder = os.path.join(coco_main_folder, part)
    yolo_part_folder = os.path.join(yolo_main_folder, part)
    os.makedirs(os.path.join(yolo_part_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_part_folder, 'labels'), exist_ok=True)

    annotations_path = os.path.join(coco_part_folder, '_annotations.coco.json')
    if os.path.isfile(annotations_path):
        process_annotations(annotations_path, os.path.join(yolo_part_folder, 'labels'), categories, cat_map)
    copy_images(coco_part_folder, os.path.join(yolo_part_folder, 'images'))

def convert_coco_to_yolo_v8(coco_main_folder, yolo_main_folder):
    if os.path.exists(yolo_main_folder):
        print(f"Error: The directory '{yolo_main_folder}' already exists. Please specify a different output directory.")
        sys.exit(1)

    categories = None
    cat_map = {}

    # Load categories and cat_map from train annotations as an example
    with open(os.path.join(coco_main_folder, 'train', '_annotations.coco.json')) as f:
        annotations = json.load(f)
        categories = annotations['categories']
        cat_map = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Process train and valid sets
    for part in ['train', 'valid']:
        process_set(coco_main_folder, yolo_main_folder, part, categories, cat_map)

    # Process test set if it exists, otherwise copy valid to test
    if os.path.isdir(os.path.join(coco_main_folder, 'test')):
        process_set(coco_main_folder, yolo_main_folder, 'test', categories, cat_map)
    else:
        test_folder_yolo = os.path.join(yolo_main_folder, 'test')
        shutil.copytree(os.path.join(yolo_main_folder, 'valid'), test_folder_yolo)

    # Generate data.yaml
    data_yaml_content = {
        'train': os.path.relpath(os.path.join(yolo_main_folder, 'train', 'images'), yolo_main_folder),
        'val': os.path.relpath(os.path.join(yolo_main_folder, 'valid', 'images'), yolo_main_folder),
        'test': os.path.relpath(os.path.join(yolo_main_folder, 'test', 'images'), yolo_main_folder),
        'nc': len(categories),
        'names': [cat['name'] for cat in categories]
    }
    
    with open(os.path.join(yolo_main_folder, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=False)


if __name__ == '__main__':
    coco_main_folder = '/home/fdias/repositorios/media/data/yolo_ptbxl_full/0'  # Path to your COCO dataset
    yolo_main_folder = '/home/fdias/repositorios/media/data/yolo_ptbxl_full_yoloformat'  # Desired path for the YOLO dataset

    convert_coco_to_yolo_v8(coco_main_folder, yolo_main_folder)
