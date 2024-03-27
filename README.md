# Time-series Forecasting
This repository contains the code for time-series forecasting using various models, such as InceptionTime, LSTM, XGBoost etc. It contains the complete pipeline from data preprocessing, model building, training, hyperparameter tuning, and evaluation.

MLFlow is used to track the experiments and log the metrics, parameters, and artifacts. Hydra is used for configuration management, and Optuna is used for hyperparameter optimization.

The code is designed to be modular and scalable. You can easily add new models, preprocessors, and data loaders!

### Data
The data used in this project is the [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) from UCI Machine Learning Repository. The data contains hourly air quality data from 12 different sites in Beijing. The data is preprocessed and aggregated to hourly/daily/weekly data for the forecasting purpose.

### Models
The following models are used for forecasting:

- XGBoost
- LSTM
- InceptionTime
- etc.


### Results
The following are the results of the forecasting models:

 - InceptionTime with daily data, past_horizon = 10, forecast_horizon = 1:

![Image](mlruns/951558179134731346/7ced2c0056484ee2840a84255cbcbaf2/artifacts/original_vs_predicted.png)

- LSTM with daily data, past_horizon = 10, forecast_horizon = 1, with Standardized data:

![Image](mlruns/393439002756371774/ac975219548c403888db88d86348667d/artifacts/original_vs_predicted.png)

- XGBoost with daily data, past_horizon = 10, forecast_horizon = 1:

![Image](mlruns/381073146271177264/1b8c9c27f2404f089532d3c71724464a/artifacts/original_vs_predicted.png)

Different windows were used for forecasting purpuse in case of XGBoost, so the plots are not comparable.

### Requirements
`requirements.yml` contains the required libraries to run the code. You can install them using the following command:
```bash
conda env create -f requirements.yml
```

To update with new libraries:
```bash
conda env export --no-builds | grep -v "prefix" > requirements.yml
```