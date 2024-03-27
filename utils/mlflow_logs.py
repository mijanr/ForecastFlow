import mlflow

def log_results(cfg, results):
    """
    Log results to mlflow

    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    results : dict
        Results dictionary    
    """
    model_name = cfg.models.arch.model_name

    # set experiment name with the model name
    mlflow.set_experiment(model_name)
    
    # station_name
    station_name = cfg.datasets.data_params.station_name

    with mlflow.start_run(run_name=station_name):
        
        # log mae, and mse
        mlflow.log_metric('mae', results['metrics']['mae'])
        mlflow.log_metric('mse', results['metrics']['mse'])

        # log window size and forecast_horizon
        mlflow.log_param('window_size', cfg.datasets.data_params.window_size)
        mlflow.log_param('horizon', cfg.datasets.data_params.forecast_horizon)

        # log target column name
        mlflow.log_param('target_column', cfg.datasets.data_params.target)

        # log granularity
        mlflow.log_param('granularity', cfg.datasets.data_params.granularity)

        # log fig as artifact
        mlflow.log_figure(results['fig'], 'original_vs_predicted.png')

        # log model name
        mlflow.log_param('model_name', model_name)

        mlflow.end_run()
        
        return

