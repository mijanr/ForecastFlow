# Time-series Forecasting
This repository contains the code for time-series forecasting using various models, such as ARIMA, SARIMA, LSTM, etc. It contains the complete pipeline from data preprocessing, model building, training, hyperparameter tuning, and evaluation.

MLFlow is used to track the experiments and log the metrics, parameters, and artifacts. Hydra is used for configuration management, and Optuna is used for hyperparameter optimization.

### Data
The data used in this project is the [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) from UCI Machine Learning Repository. The data contains hourly air quality data from 12 different sites in Beijing. The data is preprocessed and aggregated to hourly/daily/weekly data for the forecasting purpose.

### Models
The following models are used for forecasting:
- ARIMA
- SARIMA
- LSTM
- XGBoost


### Results


### Requirements
`requirements.yml` contains the required libraries to run the code. You can install them using the following command:
```bash
conda env create -f requirements.yml
```