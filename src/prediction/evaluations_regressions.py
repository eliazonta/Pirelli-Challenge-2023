from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, TheilSenRegressor
from sklearn.naive_bayes import GaussianNB
import sklearn.decomposition as decomposition
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
import os
import seaborn as sns
import pandas as pd
from custom_print import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.pipeline import make_pipeline


from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import AdaBoostRegressor

def adaboost_regression(x,y):
    print_configs("Model: AdaBoost")
    model = AdaBoostRegressor(random_state=0, n_estimators=100).fit(x, y)
    return model

def stack_regression(x,y):
    estimators = [('adaboost', AdaBoostRegressor(random_state=0, n_estimators=100)), ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.2))]
    model = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor(n_estimators=10,random_state=42)).fit(x, y)
    return model

def log_regression(x,y):
    print_configs("Model: Logistic Regression")
    model = LogisticRegression(random_state=0).fit(x, y)
    return model

def svr_regression(x,y):
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    print_configs("Model: SVR")
    model = SVR(kernel='poly', C=1.0, epsilon=0.2).fit(x, y)
    return model
    
def lin_regr(x, y):
    print_configs("Model: Linear Regression")
    model = LinearRegression().fit(x, y)
    return model

def grad_boost(x, y):
    print_configs("Model: Gradient Boosting")
    model = GradientBoostingRegressor(random_state=0).fit(x, y)
    return model

def perceptron(x, y):
    print_debugging("Model: Perceptron")
    model = Perceptron(tol=1e-3, random_state=0).fit(x, y)
    return model

def naive_bayes(x, y):
    print_debugging("Model: Naive Bayes")
    model = GaussianNB().fit(x, y)
    return model

def theilsen(x, y):
    print_debugging("Model: TheilSen")
    model = TheilSenRegressor(random_state=0).fit(x, y)
    return model

def ml_nn(x, y):
    print_configs("Model: Multi Layer Perceptron")
    model = MLPRegressor(random_state=1, max_iter=2000, activation='tanh', solver='lbfgs').fit(x, y)
    return model

def nn_linear(x, y):
    print_configs("Model: Neural Network")
    input_size = x.shape[1]
    output_size = 1
    
    layer = [
                    torch.nn.Linear(input_size, input_size // 2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(input_size // 2, output_size // 2),
                    torch.nn.ReLU(inplace=True),
                ]
    #model = torch.nn.Sequential(*layer).to('cpu')
    
    model = nn.Linear(input_size, output_size)
    model.train()
    model = model.to('cpu')
    model = model(x,y)
    return model

def eval_torch(x, y):
    model = nn_linear(x,y)
    model.eval()
    model = model.to('cuda')
    model_predictions = model(x,y)
    print(model_predictions)

def scale(x,y):
    scaler_x = preprocessing.RobustScaler()
    scaler_y = preprocessing.RobustScaler()
    x = scaler_x.fit_transform(x)
    y = y.ravel()
    return x, y, scaler_y

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
    
    # print_info("Predictions:")
    # for i in range(len(y_test)//20):
    #     print("y_test[{}]: {}, model_prediction[{}]: {}".format(i, y_test[i], i, model_prediction[i]))
    
    return model

def evaluations_on_new_data(x,y, f, desc = None):
    df_test, _ = get_data(80)
    df_test = df_test.fillna(0)
    df_test.drop(columns=['y-m-day-hour_3_rounded'], inplace=True)
    y_test = df_test['cumulative_per_day_CYCLE_ABORTED_day'].to_numpy()
    x_test , _= linearize_input(df_test)
    
    X_test, y_test, scaler = scale(x_test, y_test)
    
    if desc is not None:
        print_info(desc)
    
    X_train = x
    y_train = y
    model = f(X_train, y_train)
    train_score = model.score(X_train, y_train)
    print_info(f"Model score: {train_score}")
    model_prediction = model.predict(X_test)
    accuracy_score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, model_prediction)
    r2 = r2_score(y_test, model_prediction)
    print_info("Metrics are:\n Accuracy: {}\n MSE: {} \n R2: {}\n\n".format(accuracy_score, mse, r2))
    
    # print_info("Predictions:")
    # for i in range(len(y_test)//20):
    #     print("y_test[{}]: {}, model_prediction[{}]: {}\n\n".format(i, y_test[i], i, model_prediction[i]))
    
    return model

def linearize_input(df, side = None):
    list_status = ['CYCLE_COMPLETED', 'CYCLE_ABORTED', 'CYCLE_NOT_STARTED', 'CYCLE_COMPLETED_L', 'CYCLE_ABORTED_L', 'CYCLE_NOT_STARTED_L',
                   'CYCLE_COMPLETED_R', 'CYCLE_ABORTED_R', 'CYCLE_NOT_STARTED_R', 'cumulative_per_day_CYCLE_COMPLETED_day', 
                   'cumulative_per_day_CYCLE_ABORTED_day', 'cumulative_per_day_CYCLE_NOT_STARTED_day', 'cumulative_per_day_CYCLE_COMPLETED_L_day',
                   'cumulative_per_day_CYCLE_ABORTED_L_day', 'cumulative_per_day_CYCLE_NOT_STARTED_L_day', 'cumulative_per_day_CYCLE_COMPLETED_R_day',
                   'cumulative_per_day_CYCLE_ABORTED_R_day', 'cumulative_per_day_CYCLE_NOT_STARTED_R_day']
    df = df.drop(columns=list_status)
    print_debugging(f"Columns to drop:")
    col_name = df.columns.to_list()
    if side is None:
        #print([item for item in col_name if '_L' in item or '_R' in item])
        df.drop([item for item in col_name if '_L' in item or '_R' in item], axis=1, inplace=True)
    
    x = df.to_numpy()
    return x, df



def get_data(mach_index):
    file_dir = 'src/Data/data_per_machine/2022/processed_with_cumulatives/'
    file_list = os.listdir(file_dir)
    mach_name = [file.replace('.csv','') for file in file_list]
    file = file_dir + file_list[mach_index]
    df = pd.read_csv(file)
    current_machine = mach_name[mach_index]
    print_configs(f"Machine: {current_machine}")
    return df, current_machine

def split_tot_L_R_df(df):
    list_col = df.columns.to_list()
    list_tot_col = []
    list_L_col = []
    list_R_col = []
    for col in list_col:
        if '_L' in col:
            list_L_col.append(col)
        elif '_R' in col:
            list_R_col.append(col)
        else:
            list_tot_col.append(col)
    df_tot = df[list_tot_col]
    df_L = df[list_L_col]
    df_R = df[list_R_col]
    return df_tot, df_L, df_R

def get_x_y(df):
    cols = df.columns.to_list()
    for col in cols:
        if 'cumulative_per_day_n_barcode' in col:
            y = df[col].to_numpy()
    cols_to_drop = ['y-m-day-hour_6_rounded', 'cumulative_per_day_n_barcode',
       'cumulative_per_day_n_barcode_L', 'cumulative_per_day_n_barcode_R',
       'CYCLE_NOT_STARTED', 'CYCLE_NOT_STARTED_L', 'CYCLE_NOT_STARTED_R',
       'cumulative_per_day_CYCLE_COMPLETED',
       'cumulative_per_day_CYCLE_ABORTED',
       'cumulative_per_day_CYCLE_NOT_STARTED',
       'cumulative_per_day_CYCLE_COMPLETED_L',
       'cumulative_per_day_CYCLE_ABORTED_L',
       'cumulative_per_day_CYCLE_NOT_STARTED_L',
       'cumulative_per_day_CYCLE_COMPLETED_R',
       'cumulative_per_day_CYCLE_ABORTED_R',
       'cumulative_per_day_CYCLE_NOT_STARTED_R']
    _cols_to_drop = []
    for col in cols_to_drop:
        if col in cols:
            _cols_to_drop.append(col)
    
    df = df.drop(columns=_cols_to_drop)
    x_cols = df.columns.to_list()
    x = df.to_numpy()
    return x, y, x_cols

if __name__ == '__main__':
    dfs = []
    for i in range(20):
        df, current_machine = get_data(i)
        print_debugging(current_machine)
        dfs.append(df)
    df = pd.concat(dfs)
    
    df_tot, df_L, df_R = split_tot_L_R_df(df)
    
    x, y , x_cols= get_x_y(df_tot)
    
    # model = evaluations(x,y, log_regression, desc = "Logistic Regression")
    pca = decomposition.PCA(n_components=1)
    x_ = pca.fit_transform(x)
    
    scale_x, scale_y, scaler = scale(x,y)
    
    model_svr = evaluations(scale_x,scale_y, svr_regression, desc = "SVR")
    model_lin = evaluations(scale_x,scale_y, lin_regr, desc = "Linear Regression")
    model_gb = evaluations(scale_x,scale_y, grad_boost, desc = "Gradient Boosting")
    model_perceptron = evaluations(scale_x,scale_y, perceptron, desc = "Perceptron")
    model_naive_bayes = evaluations(scale_x,scale_y, naive_bayes, desc = "Naive Bayes")
    model_ml_nn = evaluations(scale_x,scale_y, ml_nn, desc = "Multi Layer Perceptron")
    model_stack = evaluations(scale_x,scale_y, stack_regression, desc = "Stacking")
    model_adaboost = evaluations(scale_x,scale_y, adaboost_regression, desc = "AdaBoost")
    
    
    # model_gb_ = evaluations_on_new_data(scale_x,scale_y, grad_boost, desc = "Gradient Boosting")
    
    figure = plt.subplots(10,3)
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    
    
    
    ax1.plot(scale_y, label='Actual')
    
    ax1.plot(model_gb.predict(scale_x), label='Gradient Boosting')
    ax1.legend()
    ax1.grid()
    ax1.set_xlim(0,300)
    ax2.plot(scale_y, label='Actual')
    ax2.plot(model_svr.predict(scale_x), label='SVR')
    ax2.legend()
    ax2.grid()
    ax2.set_xlim(0,300)
    ax3.plot(scale_y, label='Actual')
    ax3.plot(model_ml_nn.predict(scale_x), label='NN')
    ax3.legend()
    ax3.grid()
    ax3.set_xlim(0,300)
    figure[0].set_size_inches(20,30)
    plt.xlabel(f'PCA Components: {x_cols}')
    plt.ylabel('Cumulative Aborted Cycles per day')
    plt.savefig('src/prediction/figures/evaluations_of_models_PCA_Over_COmulative_Cycle_Aborted.png')
    plt.show()
    
    
    from lime import lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(scale_x, feature_names=x_cols, class_names=['cumulative_per_day_n_barcodes'], discretize_continuous=True)
    explained = explainer.explain_instance(scale_x[0], model_gb.predict, num_features=5)
    explainer.as_pyplot_figure()
    plt.show()
    
    
    
    # df = df.drop(columns=['Unnamed: 0'])
    # df = df.dropna()
    # df = df.drop(columns=['c_machine'])
    # df = df.drop(columns=['y-m-day-hour_3_rounded'])
    # df = df.drop(columns=['y-m-d-hour'])
    # df = df.drop(columns=['y-m-day'])
    # df = df.drop(columns=['month'])
    # df = df.drop(columns=['