import os
import seaborn as sns
import pandas as pd
from custom_print import *
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import time

class ColumnsInput:
    barcode = 'ddc_barcode'
    ipcode = 'ddc_ipcode'
    machine = 'ddc_mch_code'
    machine_side = 'ddc_mch_side'
    event = 'ddc_ev_subcode'
    time = 'ddc_ev_timestamp'
    
    
class ColumnsOutput:
    c_machine = 'c_machine' 
    event_delta_time = 'event_delta_time'
    day = 'y-m-day'
    hour = 'y-m-d-hour'
    month = 'month'
    status = 'status'

def get_list_df(dir):
    list_df = []
    mach_names = []
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir, file), low_memory=False)
            list_df.append(df)
            mach_names.append(file.replace('.csv',''))
    return list_df, mach_names

def set_barcode_cycle_label(df):
    df_label = df.sort_values(ColumnsInput.time).groupby([ColumnsInput.barcode, ColumnsInput.machine_side])[ColumnsInput.event].agg(['first', 'last']).reset_index()
    df_label['Cycle_Status'] = False
    starting_event = ["LO_LOADER_IN_PRESS", "LO_LOADER_IN_PRESS_START"]
    ending_event = ["UN_UNLOADER_OUT", "UN_FORK_OUT", "UN_UNLOADER_OUT_STOP", "UN_FORK_OUT_STOP"]
    df_label['Cycle_Status'] = df_label.apply(lambda x: 'CYCLE_COMPLETED' if x['last'] in ending_event else 'CYCLE_ABORTED' if x['first'] in starting_event else 'CYCLE_NOT_STARTED', axis=1)
    
    df = pd.merge(df, df_label[[ColumnsInput.barcode, 'Cycle_Status']], on=[ColumnsInput.barcode], how='left')
    return df

def get_time_diff(df):
    
    df[ColumnsInput.time] = pd.to_datetime(df[ColumnsInput.time])
    df = set_barcode_cycle_label(df)
    df_L = df[df[ColumnsInput.machine_side] == 'L']
    df_R = df[df[ColumnsInput.machine_side] == 'R']
    df_L = df_L.groupby([ColumnsInput.barcode, 'Cycle_Status'])[ColumnsInput.time].agg(['min', 'max']).reset_index()
    df_R = df_R.groupby([ColumnsInput.barcode, 'Cycle_Status'])[ColumnsInput.time].agg(['min', 'max']).reset_index()
    df_L= df_L.sort_values('min')
    df_R = df_R.sort_values('min')
    df_L['Between_Cycle_Time'] = df_L['min'].sub(df_L['max'].shift()).dt.total_seconds().abs()
    df_R['Between_Cycle_Time'] = df_R['min'].sub(df_R['max'].shift()).dt.total_seconds().abs()
    df_L = df_L.drop(columns=['min', 'max'])
    df_R = df_R.drop(columns=['min', 'max'])
    df = pd.concat([df_L, df_R])
    df = df.fillna(0)
    from sklearn.ensemble import IsolationForest
    
    df = pd.concat([df_L, df_R])
    df = df.fillna(0)
    isol_f = IsolationForest(random_state=42).fit(df[['Between_Cycle_Time']])
    df['Outlier'] = isol_f.predict(df[['Between_Cycle_Time']])
    df = df[df['Outlier'] == 1]
    df = df.drop(columns=['Outlier'])
    return df
    
def eval_time_per_cycle(df_list, mach_names):
    dfs = []
    for df in df_list:
        df = get_time_diff(df)
        dfs.append(df)
    df = pd.concat(dfs)
    
    mean = df.groupby('Cycle_Status')['Between_Cycle_Time'].mean()
    std = df.groupby('Cycle_Status')['Between_Cycle_Time'].std()
    n_occurences = df.groupby('Cycle_Status')['Between_Cycle_Time'].value_counts()
    
    mean = mean.drop(index='CYCLE_NOT_STARTED')
    std = std.drop(index='CYCLE_NOT_STARTED')
    figure , ax = plt.subplots()
    print_info(f"Mean time between cycles: {mean}")
    print_info(f"Std time between cycles: {std}")
    print_info(f"Number of occurences: {n_occurences}")
    bars = ax.bar(mean.index, mean.values, yerr=std.values)
    ax.bar_label(bars)
    ax.set_title('Mean time between cycles')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Cycle status')
    plt.grid()
    plt.savefig('src/prediction/figures/mean_time_between_cycles.png')
    plt.show()
    
if __name__ == '__main__':
    dir = 'src/Data/data_per_machine/2022/raw/'
    list_df, mach_names = get_list_df(dir)
    #df = set_barcode_cycle_label(test)
    eval_time_per_cycle(list_df, mach_names)
    
    
    
    # list_df = get_list_df(dir)
    # eval_time_per_cycle(list_df)