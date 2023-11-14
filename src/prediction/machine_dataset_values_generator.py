import datetime
import os
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing as preprocessing
from custom_print import *
from evaluations import *
from scipy import stats
from sklearn import metrics, preprocessing, svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
pd.set_option('display.min_rows', 30)
pd.set_option('display.max_rows',150)
pd.set_option('display.width', 1000)

# Useful classes to have easier access to data features
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
    hour_3 = 'y-m-day-hour_3_rounded'
    month = 'month'
    status = 'status'
    
def get_data(mach_index):
    file_dir = 'src/Data/data_per_machine/2022/raw/'
    file_list = os.listdir(file_dir)
    mach_name = [file.replace('.csv','') for file in file_list]
    file = file_dir + file_list[mach_index]
    df = pd.read_csv(file)
    df[ColumnsInput.time] = pd.to_datetime(df[ColumnsInput.time])
    current_machine = mach_name[mach_index]
    days = df[ColumnsOutput.day].unique()
    df[ColumnsOutput.hour] = df[ColumnsInput.time].dt.strftime('%y-%m-%d-%H')
    df['rounded_timestamp'] = df[ColumnsInput.time].dt.floor('6H')
    df['rounded_year'] = df['rounded_timestamp'].dt.year
    df['rounded_month'] = df['rounded_timestamp'].dt.month
    df['rounded_day'] = df['rounded_timestamp'].dt.day
    df['rounded_hour'] = df['rounded_timestamp'].dt.hour

    # Create a new column for split the timestamps every 3 hours
    df[ColumnsOutput.hour_3] = df['rounded_timestamp'].dt.strftime('%y-%m-%d-%H')
    df = df.drop(columns=['rounded_timestamp', 'rounded_year', 'rounded_month', 'rounded_day', 'rounded_hour'])
    df[ColumnsOutput.hour_3] = pd.to_datetime(df[ColumnsOutput.hour_3])
    print_configs(f"Machine: {current_machine}")
    return df, current_machine

def get_working_time_3h(df):
    working_time_per_3h = df.groupby([ColumnsOutput.hour_3])[ColumnsInput.time].agg(['min', 'max']).reset_index()
    working_time_per_3h['work_time'] = (working_time_per_3h['max'] - working_time_per_3h['min']).dt.total_seconds()

    working_time_per_3h_per_arm = df.groupby([ColumnsOutput.hour_3, ColumnsInput.machine_side])[ColumnsInput.time].agg(['min', 'max']).reset_index()
    working_time_per_3h_per_arm['work_time'] = (working_time_per_3h_per_arm['max'] - working_time_per_3h_per_arm['min']).dt.total_seconds()
    working_time_per_3h_per_right = working_time_per_3h_per_arm[working_time_per_3h_per_arm[ColumnsInput.machine_side] == 'R'].reset_index().drop('index', axis = 1)
    working_time_per_3h_per_right.rename(columns={'work_time':'work_time_R'}, inplace=True)
    working_time_per_3h_per_left = working_time_per_3h_per_arm[working_time_per_3h_per_arm[ColumnsInput.machine_side] == 'L'].reset_index().drop('index', axis = 1)
    working_time_per_3h_per_left.rename(columns={'work_time':'work_time_L'}, inplace=True)
    
    working_time_per_3h.drop(['min', 'max'], axis=1, inplace=True)
    working_time_per_3h_per_left.drop([ColumnsInput.machine_side,'min', 'max'], axis=1, inplace=True)
    working_time_per_3h_per_right.drop([ColumnsInput.machine_side ,'min', 'max'], axis=1, inplace=True)
        
    return working_time_per_3h, working_time_per_3h_per_left, working_time_per_3h_per_right

def get_processed_barcodes_3h(df):
    mach_tyre_per_3h = df.groupby([ColumnsOutput.hour_3])[ColumnsInput.barcode].nunique().dropna().reset_index()
    mach_tyre_per_3h_per_arm = df.groupby([ColumnsOutput.hour_3, ColumnsInput.machine_side])[ColumnsInput.barcode].nunique().reset_index()
    mach_tyre_per_3h_per_left = mach_tyre_per_3h_per_arm[mach_tyre_per_3h_per_arm[ColumnsInput.machine_side] == 'L'].reset_index().drop('index', axis = 1)
    mach_tyre_per_3h_per_right = mach_tyre_per_3h_per_arm[mach_tyre_per_3h_per_arm[ColumnsInput.machine_side] == 'R'].reset_index().drop('index', axis = 1)
    mach_tyre_per_3h_per_left.rename(columns={ColumnsInput.barcode:'n_barcode_L'}, inplace=True)
    mach_tyre_per_3h_per_right.rename(columns={ColumnsInput.barcode:'n_barcode_R'}, inplace=True)
    
    mach_tyre_per_3h_per_left.drop(ColumnsInput.machine_side, axis=1, inplace=True)
    mach_tyre_per_3h_per_right.drop(ColumnsInput.machine_side, axis=1, inplace=True)
    mach_tyre_per_3h.rename(columns={ColumnsInput.barcode:'n_barcode'}, inplace=True)
    return mach_tyre_per_3h, mach_tyre_per_3h_per_left, mach_tyre_per_3h_per_right

def set_label(df):
    df_with_labels = df.sort_values(ColumnsInput.time).groupby([ColumnsOutput.hour_3, ColumnsInput.machine_side,ColumnsInput.barcode], dropna=False)[ColumnsInput.event].agg(['first', 'last']).reset_index()
    df_with_labels[ColumnsOutput.hour_3].nunique()
    df_with_labels[ColumnsOutput.status] = False
    starting_event = ["LO_LOADER_IN_PRESS", "LO_LOADER_IN_PRESS_START"]
    ending_event = ["UN_UNLOADER_OUT", "UN_FORK_OUT", "UN_UNLOADER_OUT_STOP", "UN_FORK_OUT_STOP"]

    df_with_labels[ColumnsOutput.status] = df_with_labels.apply(lambda x: 'CYCLE_COMPLETED' if x['last'] in ending_event else 'CYCLE_ABORTED' if x['first'] in starting_event else 'CYCLE_NOT_STARTED', axis=1)
    df_with_labels.value_counts(ColumnsOutput.status)
    cycle_statuses_per_3h = df_with_labels.groupby([ColumnsOutput.hour_3], dropna=False)[ColumnsOutput.status].value_counts().unstack(fill_value=0).reset_index()

    status_names = ['CYCLE_COMPLETED','CYCLE_ABORTED', 'CYCLE_NOT_STARTED']

    for status in status_names:
        if status not in cycle_statuses_per_3h.columns:
            print(f'{status} not in columns')
            cycle_statuses_per_3h[status] = 0
    statuses_per_3h = []
    statuses_per_3h.append(cycle_statuses_per_3h.drop(['CYCLE_ABORTED', 'CYCLE_NOT_STARTED'], axis=1))
    statuses_per_3h.append(cycle_statuses_per_3h.drop(['CYCLE_COMPLETED', 'CYCLE_NOT_STARTED'], axis=1))
    statuses_per_3h.append(cycle_statuses_per_3h.drop(['CYCLE_COMPLETED', 'CYCLE_ABORTED'], axis=1))
    for i in statuses_per_3h:
        i.rename(columns={'CYCLE_COMPLETED':'count', 'CYCLE_ABORTED':'count', 'CYCLE_NOT_STARTED':'count'}, inplace=True)
    return df_with_labels , cycle_statuses_per_3h
        
def get_n_occurences_per_event(df):
    n_events = df.groupby([ColumnsOutput.hour_3], dropna=False)[ColumnsInput.event].value_counts().unstack(fill_value=0).reset_index()
    n_events_ = df.groupby([ColumnsOutput.hour_3, ColumnsInput.machine_side], dropna=False)[ColumnsInput.event].value_counts().unstack(fill_value=0).reset_index()
    n_events_left = n_events_[n_events_[ColumnsInput.machine_side] == 'L'].reset_index()    
    
    
    for t in n_events_left.columns:
        if t != ColumnsOutput.hour_3:
            n_events_left.rename(columns={t: t+'_L'}, inplace=True)
    
    n_events_right = n_events_[n_events_[ColumnsInput.machine_side] == 'R'].reset_index()
    for t in n_events_right.columns:
        if t != ColumnsOutput.hour_3:
            n_events_right.rename(columns={t: t+'_R'}, inplace=True)
    
    n_events_left.drop([ColumnsInput.machine_side + '_L', 'index_L'], axis=1, inplace=True)
    n_events_right.drop([ColumnsInput.machine_side+'_R', 'index_R'], axis=1, inplace=True)
    
    return n_events, n_events_left, n_events_right

    
    

def set_label_LR(label_cycle_status: pd.DataFrame):
    cycle_statuses_per_3h_ = label_cycle_status.groupby([ColumnsOutput.hour_3, ColumnsInput.machine_side], dropna=False)[ColumnsOutput.status].value_counts().unstack(fill_value=0).reset_index()
    cycle_statuses_per_3h_left = cycle_statuses_per_3h_[cycle_statuses_per_3h_[ColumnsInput.machine_side] == 'L'].reset_index()
    cycle_statuses_per_3h_right = cycle_statuses_per_3h_[cycle_statuses_per_3h_[ColumnsInput.machine_side] == 'R'].reset_index()

    status_names = ['CYCLE_COMPLETED','CYCLE_ABORTED', 'CYCLE_NOT_STARTED']

    for status in status_names:
        if status not in cycle_statuses_per_3h_left.columns:
            cycle_statuses_per_3h_left[status] = 0
        if status not in cycle_statuses_per_3h_right.columns:
            cycle_statuses_per_3h_right[status] = 0
    cycle_statuses_per_3h_left.drop([ColumnsInput.machine_side, 'index'], axis=1, inplace=True)
    for t in cycle_statuses_per_3h_left.columns:
        if t != ColumnsOutput.hour_3:
            cycle_statuses_per_3h_left.rename(columns={t: t+'_L'}, inplace=True)
    cycle_statuses_per_3h_right.drop([ColumnsInput.machine_side, 'index'], axis=1, inplace=True)
    for t in cycle_statuses_per_3h_right.columns:
        if t != ColumnsOutput.hour_3:
            cycle_statuses_per_3h_right.rename(columns={t: t+'_R'}, inplace=True)
    return cycle_statuses_per_3h_left, cycle_statuses_per_3h_right

def set_cumulative_label_TLR(df: pd.DataFrame):
    # Description:
    # input: dataframe of the machine
    # output: dataframe of the machine with cumulative values
    # This function is used to generate the cumulative values of the dataset
    # It generates the cumulative values of the dataset and save it in the folder src/Data/data_per_machine/2022/processed/
    # It returns the dataframe of the machine with cumulative values, resetting the values at the beginning of each day
    
    
    list_status = ['CYCLE_COMPLETED', 'CYCLE_ABORTED', 'CYCLE_NOT_STARTED', 'CYCLE_COMPLETED_L', 'CYCLE_ABORTED_L', 'CYCLE_NOT_STARTED_L', 'CYCLE_COMPLETED_R', 'CYCLE_ABORTED_R', 'CYCLE_NOT_STARTED_R']
    
    for i in list_status:
        df[f'cumulative_per_day_{i}_day'] = df.groupby(df[ColumnsOutput.hour_3].dt.floor('D'))[i].cumsum()    
    return df

def set_cumulative_barcode_TLR(label_cycle_status: pd.DataFrame):
    pass

def set_machine_df(mach_index):
    # Description:
    # input: index of the machine to analyse
    # output: dataframe of the machine and the name of the machine
    # This function is used to generate the dataset for each machine
    # It generates the dataset for each machine and save it in the folder src/Data/data_per_machine/2022/processed/
    # It returns the dataframe of the machine and the name of the machine
    # The dataset combine int/float data every 3 hour per machines, the generated columns are:
    # - n_barcode: number of processed barcode every 3 hours
    # - n_barcode_L: number of processed barcode of the left arm every 3 hours
    # - n_barcode_R: number of processed barcode of the right arm every 3 hours
    # - work_time: working time of the machine every 3 hours
    # - work_time_L: working time of the left arm of the machine every 3 hours
    # - work_time_R: working time of the right arm of the machine every 3 hours
    # - CYCLE_COMPLETED: number of completed cycle every 3 hours
    # - CYCLE_ABORTED: number of aborted cycle every 3 hours
    # - CYCLE_NOT_STARTED: number of not started cycle every 3 hours
    # - CYCLE_COMPLETED_L: number of completed cycle of the left arm every 3 hours
    # - CYCLE_ABORTED_L: number of aborted cycle of the left arm every 3 hours
    # - CYCLE_NOT_STARTED_L: number of not started cycle of the left arm every 3 hours
    # - CYCLE_COMPLETED_R: number of completed cycle of the right arm every 3 hours
    # - CYCLE_ABORTED_R: number of aborted cycle of the right arm every 3 hours
    # - CYCLE_NOT_STARTED_R: number of not started cycle of the right arm every 3 hours
    # - n_events per each type of event, L and R as well: number of occurences of each event every 3 hours
    
    df, current_machine = get_data(mach_index)
    df_n_events = get_n_occurences_per_event(df)    
    
    df_work_time = get_working_time_3h(df)
    df_barcodes = get_processed_barcodes_3h(df)
    df_machine_ = pd.merge(df_barcodes[0], df_barcodes[1], on=ColumnsOutput.hour_3, how='left')
    df_machine_ = pd.merge(df_machine_, df_barcodes[2], on=ColumnsOutput.hour_3, how='left')    
    for i in df_work_time:
        df_machine_ = pd.merge(df_machine_, i, on=ColumnsOutput.hour_3, how='left')
    
    df_label, df_div_by_label = set_label(df)
    df_machine_ = pd.merge(df_machine_, df_div_by_label, on=ColumnsOutput.hour_3, how='left')
   
    df_label_LR = set_label_LR(df_label)
    for i in df_label_LR:
        df_machine_ = pd.merge(df_machine_, i, on=ColumnsOutput.hour_3, how='left')

    # adding cumulative values:
    df_machine_ = set_cumulative_label_TLR(df_machine_)
    for i in df_n_events:
        df_machine_ = pd.merge(df_machine_, i, on=ColumnsOutput.hour_3, how='left')
    df_machine_.fillna(0, inplace=True)
    return df_machine_, current_machine
    
def save_df(df, location):
    df.to_csv(location, index=False)
    
def analyse_generated_dataset(mach_index):
    
    # Description:
    # input: index of the machine to analyse
    # output: plots of the distribution of each feature and the correlation between each feature and the target variable
    # This function is used to analyse the generated dataset
    # It plots the distribution of each feature and the correlation between each feature and the target variable
    # It show continues plot until the user press a key, then it closes the plot and open the next one
    
    
    file_dir = 'src/Data/data_per_machine/2022/processed/'
    file_list = os.listdir(file_dir)
    mach_name = [file.replace('.csv','') for file in file_list]
    file = file_dir + file_list[mach_index]
    df = pd.read_csv(file)
    current_machine = mach_name[mach_index]
    print_configs(f"Machine: {current_machine}")
    
    df.drop(columns=['y-m-day-hour_3_rounded'], inplace=True)
    col_name = df.columns.to_list()
    df.drop([item for item in col_name if '_L' in item or '_R' in item], axis=1, inplace=True)    
    df.drop(columns=['n_barcode', 'work_time', 'CYCLE_COMPLETED', 'CYCLE_NOT_STARTED'], inplace=True)
    
    col_name = df.columns.to_list()
    for t in col_name:
        sns.displot(df[t])
        sns.jointplot(x=t, y="CYCLE_ABORTED", data=df, kind="reg")
        sns.lmplot(x=t, y="CYCLE_ABORTED", data=df)
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time
        plt.close('all')
    

if __name__ == "__main__":
    for i in range(len(os.listdir('src/Data/data_per_machine/2022/raw/'))):
        df_machine, current_machine = set_machine_df(i)
        save_df(df_machine, 'src/Data/data_per_machine/2022/processed/'+ current_machine + '.csv')
    df_machine, current_machine = set_machine_df(0)    
    analyse_generated_dataset(0)
    