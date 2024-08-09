import numpy as np
import wfdb

def array2wfdb(ecg_data, 
               record_name, 
               comments,
               save_path,
               fs,
               fmt=['16']*12,
               units=['mV']*12,
               sig_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
               ):

    wfdb.wrsamp(record_name, 
                write_dir=save_path,
                fs=fs, 
                units=units, 
                sig_name=sig_names, 
                p_signal=ecg_data, 
                fmt=fmt,
                comments=comments)