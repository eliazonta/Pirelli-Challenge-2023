import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import torch
import torch.nn as nn
from custom_print import *
#class models 
from sklearn import svm
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestClassifier, StackingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  Perceptron, SGDClassifier, TheilSenRegressor)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, LinearSVC, LinearSVR


def stack_class(x,y):
    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('svr', make_pipeline(LinearSVC(dual="auto", random_state=42)))]
    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    model.fit(x, y)
    return model

def svm_class(x,y):
    print_configs("Model: SVM")
    model = svm.SVC()
    model.fit(x, y)
    return model

def sgd_class(x,y):
    print_configs("Model: SGD")
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    model.fit(x, y)
    return model

def gp_class (x,y):
    print_configs("Model: Gaussian Process")
    kernel = 1.0 * RBF(1.0)
    model = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(x, y)
    return model

def evaluations(x,y, f, desc = None):
    
    if desc is not None:
        print_info(desc)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)    
    
    model = f(X_train, y_train)
    train_score = model.score(X_train, y_train)
    print_info(f"Model score: {train_score}")
    model_prediction = model.predict(X_test)
    accuracy_score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, model_prediction)
    r2 = r2_score(y_test, model_prediction)
    print_info("Metrics are:\n Accuracy: {}\n MSE: {} \n R2: {}\n\n".format(accuracy_score, mse, r2))
    
    print_info('\n', classification_report(y_test, model_prediction))
    
    return model

if __name__ == '__main__':
    
    df = pd.read_csv("src/Data/data_per_machine/2022/all_mach_clusters_selected.csv")
    df =  df.drop(['cumulative_per_day_CYCLE_COMPLETED_day'], axis=1)
    df = df.drop(['cumulative_per_day_CYCLE_ABORTED_day'], axis=1)
    df = df.drop(['cumulative_per_day_CYCLE_COMPLETED_L_day'], axis=1)
    df = df.drop(['cumulative_per_day_CYCLE_COMPLETED_R_day'], axis=1)
    df = df.drop(['y-m-day-hour_3_rounded', 'machine'], axis=1)
    df = df.fillna(0)
    y = df['cluster_label'].to_numpy()
    x = df.drop(['cluster_label','cluster_label_R', 'cluster_label_L'], axis=1).to_numpy()
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model_svm = evaluations(x,y, svm_class, "SVM")
    model_sgd = evaluations(x,y, sgd_class, "SGD")
    model_stack = evaluations(x,y, stack_class, "Stacking model")
    
    
    figure, ax = plt.subplots(1,2)
    sns.heatmap(confusion_matrix(y_test, model_svm.predict(x_test)), annot=True, ax=ax[0])
    sns.heatmap(confusion_matrix(y_test, model_sgd.predict(x_test)), annot=True, ax=ax[1])
    
    plt.show()