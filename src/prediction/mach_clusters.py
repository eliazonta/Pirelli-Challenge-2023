import multiprocessing
import os

import pandas as pd
from custom_print import *
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, mean_squared_error,
                             silhouette_score)


class ColumnsInput:
    barcode = 'ddc_barcode'
    ipcode = 'ddc_ipcode'
    machine = 'ddc_mch_code'
    machine_side = 'ddc_mch_side'
    event = 'ddc_ev_subcode'
    time = 'ddc_ev_timestamp'

class Machines():
    def __init__(self, dir) -> None:
        self._dir = dir
        self.file_list = os.listdir(self._dir)
        
    def time_event_per_machine(self, mach_index):
        file = self._dir + self.file_list[mach_index]
        df = pd.read_csv(file)
        
        df[ColumnsInput.time] = pd.to_datetime(df[ColumnsInput.time])
        df_R = df[df[ColumnsInput.machine_side] == 'R']
        df_L = df[df[ColumnsInput.machine_side] == 'L']
        df_R['event_time'] = df_R[ColumnsInput.time].diff().dt.total_seconds().fillna(0)
        df_L['event_time'] = df_L[ColumnsInput.time].diff().dt.total_seconds().fillna(0)
        df = pd.concat([df_R, df_L])
        print_info("file\n",df['event_time'].describe())
        z = df['event_time'].mean() + 3*df['event_time'].std()
        df = df.drop(df[df['event_time'] > z].index)
                
        return df
    
    def single_cluster_eval(self, df, n):
        
        print_info(f"Eval number of clusters: {n}")
        model = KMeans(n_clusters=n, n_init="auto", max_iter=100, random_state=42)
        label = model.fit_predict(df[['event_time']])
        curr_sil_score = silhouette_score(df[['event_time']], label)
        curr_mse = silhouette_score(df[['event_time']], label)
        
        return curr_sil_score, curr_mse
    
    def n_cluster_eval(self):
        n_cores = multiprocessing.cpu_count()
        df_ = []
        for i in range(3):
            df_.append(self.time_event_per_machine(i))
        df = pd.concat(df_)
        df = df.fillna(0)
        df = df[10000:65000]
        sil_score = []
        mse = []
        
        max = 20
        if max > n_cores:
            max = n_cores
        
        sil_score, mse = zip(*Parallel(n_jobs=n_cores)(delayed(self.single_cluster_eval)(df, i) for i in range(2, max)))
        
        # for i in range (2, 10):
        #     print_info(f"Eval number of clusters: {i}")
        #     model = KMeans(n_clusters=i, n_init="auto", max_iter=100, random_state=42)
        #     label = model.fit_predict(df[['event_time']])
        #     curr_sil_score = silhouette_score(df[['event_time']], label)
        #     curr_mse = silhouette_score(df[['event_time']], label)
        #     sil_score.append(curr_sil_score)
        #     mse.append(curr_mse)
            
        figure, ax = plt.subplots(1,2, figsize=(20,5))
        ax[0].plot(range(2,max), sil_score)
        ax[0].set_title('Silhouette Score n clusters of event time')
        ax[0].set_xlabel('Number of clusters')
        ax[0].set_ylabel('Score')
        ax[0].set_xticks(range(0,max))
        ax[0].grid()
        ax[1].plot(range(2,max), mse)
        ax[1].set_title('MSE n clusters of event time')
        ax[1].set_xlabel('Number of clusters')
        ax[1].set_ylabel('Score')
        ax[1].set_xticks(range(0,max))
        plt.grid()
        plt.savefig('src/prediction/figures/n_cluster_eval_on_time.png')
        plt.show()
        
        figure, ax = plt.subplots(1,2, figsize=(20,5))
        ax[0].plot(range(2,10), sil_score[0:8])
        ax[0].set_title('Silhouette Score n clusters of event time')
        ax[0].set_xlabel('Number of clusters')
        ax[0].set_ylabel('Score')
        ax[0].set_xticks(range(2,10))
        ax[0].grid()
        ax[1].plot(range(2,10), mse[0:8])
        ax[1].set_title('MSE n clusters of event time')
        ax[1].set_xlabel('Number of clusters')
        ax[1].set_ylabel('Score')
        ax[1].set_xticks(range(0,10))
        plt.grid()
        plt.savefig('src/prediction/figures/n_cluster_ZOOMED_eval_on_time.png')
        plt.show()
        
    def kmeans_time_events(self):
        df_ = []
        for i in range(len(self.file_list)):
            df = self.time_event_per_machine(i)
            model = KMeans(n_clusters=6, n_init="auto", max_iter=100, random_state=42).fit(df[['event_time']])
            label = model.fit_predict(df[['event_time']])
            centroids = model.cluster_centers_
            figure, ax = plt.subplots(figsize=(10,5))
            y = df['event_time']
            ax.scatter(y, y, c=label, s=50, cmap='viridis')
            ax.scatter(centroids[:,0], centroids[:,0], c='red', s=50)
            ax.set_title(f'Kmeans event time on machine {self.file_list[i].replace(".csv", "")}')
            ax.set_xlabel('Event time')
            ax.set_ylabel('Event time')
            plt.grid()
            plt.savefig(f'src/prediction/figures/kmeans/kmeans_time_events_machine_{i}.png')
            plt.close()
            df['time:_cluster_label'] = label
            save_df = df.to_csv(f'src/Data/data_per_machine/2022/raw/clustered/kmeans_time_events_machine_{self.file_list[i]}')
        
        pass
        
if __name__ == '__main__':
    dir = 'src/Data/data_per_machine/2022/raw/'
    mach = Machines(dir)
    mach.kmeans_time_events()
    