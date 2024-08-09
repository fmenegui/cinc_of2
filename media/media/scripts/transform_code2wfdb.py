'''
    code15:'/mnt/experiments2/felipe.dias/code15_processed'
'''

from media.helpers.array2wfdb import array2wfdb
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

fs = 4096./10.

def get_diagnosis(row):
    dx_classes = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
    dx_list = [dx for dx in dx_classes if row.get(dx, False)]
    
    if row.get('normal_ecg', False):
        return 'Normal'
    elif not dx_list:
        return 'Other'
    else:
        return ', '.join(dx_list)
    
    
def process_ecg(hdf5_path, csv_path, output_base):
    metadata = pd.read_csv(csv_path, sep='\t')
    output_base_img = os.path.join(output_base, 'img')
    os.makedirs(output_base_img, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as hdf:
        for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
            try:
                folder_idx = idx // 1000
                folder_path = os.path.join(output_base_img, f'batch_{folder_idx}')
                os.makedirs(folder_path, exist_ok=True)
                
                # ECG
                ecg_data = hdf['1d'][idx,:]
                
                # WFDB filename
                record_name = f"{row['exam_id']}"
                metadata.loc[idx,'wfdb_file'] = os.path.join('img', f'batch_{folder_idx}', record_name)
                
                # Include Metadata to .hea file
                comments = [
                    f"#Age: {row['age']}",
                    f"#Sex: {'Male' if row['is_male'] else 'Female'}",
                    f"#Height: Unknown",
                    f"#Weight: Unknown",
                    "#Dx: " + get_diagnosis(row)
                ]
                
                # Write WFDB file
                array2wfdb(ecg_data, 
                        record_name, 
                        comments,
                        save_path=folder_path,
                        fs=fs                       
                        )
                metadata.loc[idx,'transform_array2wfdb_success'] = True
            except:
                metadata.loc[idx,'transform_array2wfdb_success'] = False
                print(f'Error in index {idx}')
                
        print(os.path.join(output_base, csv_path.split('/')[-1]))
        metadata.to_csv(os.path.join(output_base, csv_path.split('/')[-1]), index=False)
            
if __name__ == '__main__':
    process_ecg('/mnt/experiments2/felipe.dias/code15_processed/data.hdf5',
                '/mnt/experiments2/felipe.dias/code15_processed/exams.csv',
                '/mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_WFDB_FILES/CODE15'
                )