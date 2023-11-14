# SOURCE FILES
## Introduction
This folder contains all the source files for the project. The source files are divided into data, data_generqator, notebooks, OLD, prediction, utils.
## Data
Contains al the raw and preprocessed data, divide per year. The preprocessed data were generated by the prediction/machine_dataset_values_generator.py script.
## Data_generator
[DEPRECATED] Generate random data
## Notebooks
Contains all the data exploration notebooks, in particular are relevant:
- Data_Exploration_year_2022.ipynb: contains the data exploration of the raw data
- Data_per_Machine_exploration.ipynb: contains the data exploration of the preprocessed data per machine, most relvant for the model creation.
- Others: contains the data exploration of the preprocessed data per year, not relevant for the model creation.

## OLD
Contains old scripts, not relevant for the model creation.

## Prediction
Contains all the scripts for the model creation and the prediction. The most relevant are:
- machine_dataset_values_generator.py: generate the preprocessed data per machine
- evaluation.py: contains the models used and tested, the data preprocessing, tranformation and fitting to the models, model training, evaluation metrics, prediction and plots.
- WORK in PROGRESS: evaluation should be complted with more graph and plots, ideally use [lime](https://lime-ml.readthedocs.io/en/latest/) to explain the model prediction.