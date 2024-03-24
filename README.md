# Time-series Forecasting
This repository contains the code for time-series forecasting using various models, such as ARIMA, SARIMA, LSTM, etc. It contains the complete pipeline from data preprocessing, model building, training, hyperparameter tuning, and evaluation.

MLFlow is used to track the experiments and log the metrics, parameters, and artifacts. Hydra is used for configuration management, and Optuna is used for hyperparameter optimization.


### Requirements
`requirements.yml` contains the required libraries to run the code. You can install them using the following command:
```bash
conda env create -f requirements.yml
```