'''
    incor:'/mnt/raw2/felipe/datasets/ecgia_dataset_1d'
'''
from media.helpers.array2wfdb import array2wfdb
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import re
import hashlib
import secrets

fs = 1000

def generate_salt(length=16): return secrets.token_bytes(length)
def hash_exam_id(exam_id, salt=b'fixed_salt'):
    if salt is None: salt = generate_salt() 
    hash_input = salt + exam_id.encode('utf-8') 
    hash_object = hashlib.sha256(hash_input)
    hashed_id = hash_object.hexdigest()
    return hashed_id, salt


def parse_age(age_str):
    try:
        years_match = re.search(r'(\d+)a', age_str)
        months_match = re.search(r'(\d+)m', age_str)
        days_match = re.search(r'(\d+)d', age_str)

        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        days = int(days_match.group(1)) if days_match else 0

        age_in_years = years + months / 12 + days / 365
        return age_in_years
    except:
        return 'Unknown'


def clean_dx(s):
    cleaned = re.sub(r'\s+,', ',', s)
    if cleaned.endswith(','): cleaned = cleaned[:-1]
    return cleaned
        
        
def process_ecg(hdf5_path, csv_path, output_base):
    metadata = pd.read_csv(csv_path)
    output_base_img = os.path.join(output_base, 'img')

    with h5py.File(hdf5_path, 'r') as hdf:
        for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):            
            # if idx > 5: break
            # Folder
            folder_idx = idx // 1000
            folder_path = os.path.join(output_base_img, f'batch_{folder_idx}')
            os.makedirs(folder_path, exist_ok=True)
            
            # ECG
            ecg_data = hdf['dataset'][idx,:]
            
            # WFDB filename
            hashed_id, used_salt = hash_exam_id(f"{row['ID_EXAME']}")
            record_name = hashed_id
            metadata.loc[idx,'new_id'] = record_name
            metadata.loc[idx,'wfdb_file'] = os.path.join('img', f'batch_{folder_idx}', record_name)
            
            # Include Metadata to .hea file
            age = parse_age(row['IDADE'])
            dx = row['LAUDO'].replace('|', ',').replace('.','')
            dx = clean_dx(dx)
            comments = [
                f"#Age: {age}",
                f"#Sex: {'Male' if row['SEXO']=='MASCULINO' else 'Female'}",
                f"#Height: Unknown",
                f"#Weight: Unknown",
                f"#Dx: {dx}" 
            ]
            
            # Write WFDB file
            array2wfdb(ecg_data, 
                       record_name, 
                       comments,
                       save_path=folder_path,
                       fs=fs                       
                       )
        print(os.path.join(output_base, csv_path.split('/')[-1]))
        metadata.to_csv(os.path.join(output_base, csv_path.split('/')[-1]), index=False)
            
if __name__ == '__main__':
    process_ecg('/mnt/raw2/felipe/datasets/ecgia_dataset_1d/data.h5',
                '/mnt/raw2/felipe/datasets/ecgia_dataset_1d/ecgia_xml_kfold_original_index_no_duplicates.csv',
                '/mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_WFDB_FILES/INCORDB'
                )