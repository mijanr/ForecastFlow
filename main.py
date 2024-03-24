import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(version_base='1.3.2', config_path="configs", config_name="main_config.yaml")    
def main(cfg: DictConfig) -> None:
    dataset = hydra.utils.instantiate(cfg.datasets.model_params)

    # load train and test data
    X_train, y_train, X_test, y_test = dataset.train_test_split(
        df=dataset.process_data(),
        **cfg.datasets.data_params
    )
    return
    
if __name__ == "__main__":  
    main()