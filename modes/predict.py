import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch import nn
import torch
from  torchvision.utils import save_image


import logging



# Hydra and lightning imports
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch import loggers as pl_loggers
from hydra.core.hydra_config import HydraConfig
import hydra.utils as hy_utils

from pathlib import Path

import pandas as pd


log = logging.getLogger(__name__)

# Hydra config
hydra_config = HydraConfig.get()

def predict(cfg):
    log.info("Predicting...")


    ### ===================== ###
    ### --- DATASET SETUP --- ###
    ### ===================== ###

    # Load main dataset
    dataset = hy_utils.instantiate(cfg.dataset, **cfg.dataset.test)
    
    
    ### ======================== ###
    ### --- DATALOADER SETUP --- ###
    ### ======================== ###

    # Load main dataloader
    predict_dataloader = hy_utils.instantiate(cfg.dataloader.test, dataset=dataset)


    ### ====================== ###
    ### --- TRAINING SETUP --- ###
    ### ====================== ###

    # --- Define logger and checkpoint parameters for the trainer

    # # Info from the hydra config
    dir_to_save = hydra_config.runtime.output_dir
    job_name = hydra_config.job.name

    # Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_to_save, name="", version="")
    
    # # Redirect the lightning logs to the hydra logs
    logger = logging.getLogger("lightning.pytorch")
    logger.addHandler(logging.FileHandler(dir_to_save / Path(f"{job_name}.log")))


    ### ===================== ###
    ### --- TRAINING LOOP --- ###
    ### ===================== ###

    # Define
    trainer = pl.Trainer(logger=tb_logger, **cfg.trainer)


    # Get the checkpoint path
    training_path = Path(cfg.checkpoint)
    try:
        checkpoint = next(training_path.rglob("best*.ckpt"))
    except StopIteration:
        checkpoint = next(training_path.rglob("last*.ckpt"))
    
    if checkpoint is None:
        log.error("No checkpoint found!")
        return
    

    # Get litmodule parameters
    params_to_override = dict(cfg.litmodule)
      
    # Remove the target
    params_to_override.pop("_target_")
    
    # Log the parameters to override
    log.info("Parameters to override in test:")
    log.info(params_to_override)
    
    # Load model from checkpoint
    lightning_model = hy_utils.get_class(cfg.litmodule._target_).load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location=None,
            **params_to_override
            
        )
    lightning_model.eval()

    # Test the model
    predictions = trainer.predict(model=lightning_model, dataloaders=predict_dataloader, ckpt_path=checkpoint)
    
    # --- Save predictions
    # print(predictions.shape)
    
    # Create output folder to save the images
    dataset_name = hydra_config.runtime.choices.dataset
    output_folder = training_path / Path(f"predictions_{dataset_name}")
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Save the predictions
    for i, prediction in enumerate(predictions):
        print(prediction.shape)
        save_image(prediction, output_folder / Path(f"prediction_{i}.png"))    
        
    log.info("Prediction finished!")
        



    