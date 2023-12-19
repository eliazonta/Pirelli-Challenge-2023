# Pirelli - HIT Industrial AI Challenge 2023 

This code repository contains all the scripts/notebooks used during the 12 weeks research based challenge.
Using statistical inference and machine learning to analyse and forecast the curing machines anomalies.

## Table of content
- [Data](#DATA)
    - data per ipcode
    - data per machine
- [Data generator](#GEN)
- [Notebooks](#NB)
- [Prediction](#PRED)
- [Authors](#AUTHORS)

## Data <a name="DATA"></a>
Contains all the preprocessed data, divided by year. 
- data per ipcode (type)
- data per machine

## Data generator <a name="GEN"></a>
Synthetic data generator

## Notebooks <a name="NB"></a>
Contains all the data exploration notebooks, in particular:
- Data_Exploration_year_2022.ipynb: raw data exploration
- Machine_exploration.ipynb: contains the data exploration of the preprocessed data per machine, most relvant for the model creation.
- IPcode_exploration: outliers detection and analysis
- Others: contains the data exploration of the preprocessed data per year, not relevant for the model creation.

## Prediction <a name="PRED"></a>
Contains all the scripts for the model creation and the prediction. The most relevant are:
- machine_dataset_values_generator.py: generate the preprocessed data per machine
- evaluation.py: contains the models used and tested, the data preprocessing, tranformation and fitting to the models, model training, evaluation metrics, predictions, plots and model explaination with [lime](https://lime-ml.readthedocs.io).

## Authors <a name="AUTHORS"></a>
- Anna Elise Høfde
- Elia Zonta
- Erik Nielsen
- Usama Zafar