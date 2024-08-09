import pandas as pd
import os

def save_timers(time_dict, save_dir):
    del time_dict['start']
    timers_df = pd.DataFrame(time_dict, index=[0]).T
    timers_df.reset_index(inplace=True)
    timers_df.columns = ['Timers', 'Time (s)']
    timers_df.to_csv(os.path.join(save_dir, 'timers.csv'), index=False)
    return os.path.join(save_dir, 'timers.csv')