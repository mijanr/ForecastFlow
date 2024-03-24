import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(version_base='1.3.2', config_path="configs", config_name="main_config.yaml")    
def main(cfg: DictConfig) -> None:
    dataset = hydra.utils.instantiate(cfg.datasets.model_params)
    
if __name__ == "__main__":  
    main()