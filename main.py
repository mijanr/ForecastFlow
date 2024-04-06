import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from utils.mlflow_logs import log_results
from utils.plot_results import plot_predictions

@hydra.main(version_base='1.3.2', config_path="configs", config_name="main_config.yaml")    
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.datasets.DataLoader)

    if cfg.datasets.data_params.for_tsai:
        params = cfg.datasets.data_params
        # load tsai data
        X, y, splitter, tfms, batch_tfms = dataset.get_tsai_data(**params)

        # load the model
        model = instantiate(cfg.models.model)

        # params for training
        params = {
            'X': X,
            'y': y,
            'splits': splitter,
            'tfms': tfms,
            'batch_tfms': batch_tfms,
            'arch': cfg.models.arch.model_name
        }
        # extend the params with cfg.training_params
        params.update(cfg.training_params)

        # train the model
        model.train_model(**params)

        # evaluate the model
        eval_params = {
            'arch': cfg.models.arch.model_name,
            'X': X,
            'y': y,
            'splits': splitter
        }
        outputDict = model.evaluate_model(**eval_params)

        # get fig
        fig = plot_predictions(cfg.models.arch.model_name, outputDict['target'], outputDict['preds'])

        outputDict['fig'] = fig
        
        # log results to mlflow
        log_results(cfg, outputDict)

    else:
        # load train and test data
        X_train, y_train, X_test, y_test = dataset.train_test_split(
            df=dataset.process_data(),
            **cfg.datasets.data_params
        )
        if cfg.models.arch.model_name == 'XGB':
            model = instantiate(cfg.models.model)

            # params
            params = {
                'X_train': X_train,
                'y_train': y_train,
                'arch': cfg.models.arch.model_name
            }

            # train the model
            model.train_model(**params)

            # evaluate the model
            eval_params = {
                'arch': cfg.models.arch.model_name,
                'X_test': X_test,
                'y_test': y_test
            }
            outputDict = model.evaluate_model(**eval_params)

            # get fig
            fig = plot_predictions(cfg.models.arch.model_name, outputDict['target'], outputDict['preds'])
            outputDict['fig'] = fig

            # log results to mlflow
            log_results(cfg, outputDict)

        else:
            return

    return
    
if __name__ == "__main__":  
    main()