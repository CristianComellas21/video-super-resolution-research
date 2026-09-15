import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.config_store import ConfigStore

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from pathlib import Path
from enum import Enum
from utils.hydra.resolvers import eval_expression, parse_to_type

import logging 
log = logging.getLogger(__name__)


##### ---------------- ######
##### === Aux Func === ######
##### ---------------- ######



OmegaConf.register_new_resolver("eval", eval_expression)
OmegaConf.register_new_resolver("parse", parse_to_type)

def config_to_str(cfg) -> None:
    return (
            f"""
            ========= CONFIG USED =========
            {OmegaConf.to_yaml(cfg)}
            ================================
            """
            )




    # print("========= CONFIG USED =========\n") 
    # print(OmegaConf.to_yaml(cfg))
    # print("===============================\n\n")
        
    
def import_mode_and_return_function(name: str):
    
    if name is None:
        return None
    
    mod = __import__(f"modes.{name}", fromlist=[None])
    try:
        return getattr(mod, name)
    except AttributeError:
        return None

##### ---------------- ######
##### === Main App === ######
##### ---------------- ######


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg) -> None:
    
    log.debug(config_to_str(cfg))    

    hydra_config = HydraConfig.get()
    mode_name = hydra_config.runtime.choices.get("mode", None)
    mode_function = import_mode_and_return_function(mode_name)
    
    if mode_function is not None:
        mode_function(cfg)
    else:
        raise NotImplementedError(f"Mode {mode_name} not implemented.")

if __name__ == "__main__":
    my_app()
