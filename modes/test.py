import lightning.pytorch as pl
from torch import nn
import torch

import logging

from skimage import io
import numpy as np

# Hydra and lightning imports
from lightning.pytorch import loggers as pl_loggers
from hydra.core.hydra_config import HydraConfig
import hydra.utils as hy_utils

from pathlib import Path

import pandas as pd
import utils.image.torch_dct as torch_dct
import inspect

import subprocess, os


log = logging.getLogger(__name__)

# Hydra config
hydra_config = HydraConfig.get()

def test(cfg):
    log.info("Testing...")
    
    print(cfg.dataloader.test)
    

    ### ===================== ###
    ### --- DATASET SETUP --- ###
    ### ===================== ###


    # -----------------------------------------------------------------------------------
    # # TODO: remove this selection of gpu and path used for testing
    # def get_most_free_gpu():
    #     out = subprocess.check_output([
    #         "nvidia-smi",
    #         "--query-gpu=memory.free",
    #         "--format=csv,nounits,noheader"
    #     ]).decode().strip().splitlines()
    #     free = [int(x) for x in out]  # MB libres por GPU física
    #     best = max(range(len(free)), key=lambda i: free[i])
    #     return best, free

    # best_gpu, _ = get_most_free_gpu()
    
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)  # dentro del proceso solo existirá esa GPU como cuda:0

    
    # cfg.trainer.accelerator="gpu"
    # cfg.trainer.devices=1

    # if hydra_config.runtime.choices.path == "jlserver":
    #     cfg.dataset.root = "/home/cristian/phd/bm3d/data/UIB_SINGLE_IMAGE_GRAY/test"

    # log.info(f"Dataset path chosen: {cfg.dataset.root}")
    # log.info(f"Using GPU: {cfg.trainer.devices}")
    
    # ------------------------------------------------------------------------------------
    
    
    # Load main dataset
    dataset = hy_utils.instantiate(cfg.dataset, **cfg.dataset.test)
    
    
    ### ======================== ###
    ### --- DATALOADER SETUP --- ###
    ### ======================== ###

    # Load main dataloader
    test_dataloader = hy_utils.instantiate(cfg.dataloader.test, dataset=dataset)

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

    # Define trainer
    trainer = pl.Trainer(logger=tb_logger, **cfg.trainer)

    # Get litmodule parameters
    params_to_override = dict(cfg.litmodule)
    
    # Remove the target
    params_to_override.pop("_target_")
    
    # Log the parameters to override
    log.info("Parameters to override in test:")
    log.info(params_to_override)



    # Checkpoint mode
    checkpoint_mode = cfg.get("checkpoint_mode", None)
    
    # Get the checkpoint path
    if cfg.checkpoint is not None:
        
        training_path = Path(cfg.checkpoint)
        if checkpoint_mode is not None:
            checkpoint = next(training_path.rglob(f"*{checkpoint_mode}*.ckpt"))
        else:
            try:
                checkpoint = next(training_path.rglob("best*.ckpt"))
            except StopIteration:
                checkpoint = next(training_path.rglob("last*.ckpt"))

        # Load model from checkpoint
        lightning_model = hy_utils.get_class(cfg.litmodule._target_).load_from_checkpoint(
                checkpoint_path=checkpoint,
                map_location="cpu",
                **params_to_override
                )
    
    else:
        checkpoint = None
        training_path = Path(dir_to_save) 
    
    
        if cfg.get("test_external_model", False):
            log.error("No checkpoint found, but a state dict is provided. The state dict will be loaded directly into the model.")
            
            
            # Instantiate the model
            model = hy_utils.instantiate(cfg.model)
            
            # Update the training path
            training_path = Path(dir_to_save).parent / Path(inspect.getfile(model.__class__)).stem
            
            # Create the lightning model with the model
            # and the parameters to override
            lightning_model = hy_utils.instantiate(cfg.litmodule, model=model, **params_to_override)
            
            
            # Some models do not have a state dict (They are just a function)
            # So we need to check if the state dict is not None
            # before loading it
            if cfg.get('state_dict', None) is not None:
                
                
                # TODO: Remove this when this is not needed
                # Handle very specific case of DCT2Net which saves a parameter in state_dict (very weird and specific)
                
                if cfg.model._target_ == "models.DCT2Net.Model":
                    # Get the state dict
                    state_dict = torch.load(cfg.state_dict, map_location="cpu")
                    
                    # Load the Pm1 parameter into the model
                    lightning_model.model.Pm1.data = state_dict.view(cfg.model.patch_size**2, cfg.model.patch_size**2)
                    
                else:
                    
                    
                    log.info(f"Loading state dict from {cfg.state_dict}...")
                    
                    # Load the state dict
                    state_dict = torch.load(cfg.state_dict, map_location="cpu")
                    
                    if isinstance(state_dict, dict):
                        lightning_model.model.load_state_dict(state_dict, strict=False)
                    
                    elif isinstance(state_dict, nn.Module):
                        lightning_model.model = state_dict
                            
                    
                log.info("State dict loaded successfully.")
            
            
            
        else:
            log.error("No checkpoint found!")
            lightning_model = hy_utils.instantiate(cfg.litmodule, **params_to_override)
            

    lightning_model.eval()

    
    # Test the model
    metrics = trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path=checkpoint)
    
    # Create trainig path if it does not exist
    training_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    dataset_name = dataset.name if hasattr(dataset, "name") else hydra_config.runtime.choices.dataset
    metrics_name = f"test_metrics_{dataset_name}.csv"
    
    # If the full test metrics are not available, save the metrics from the test method
    if not hasattr(lightning_model, "test_metrics"):
        df = pd.DataFrame(metrics)
        df.to_csv(training_path / Path(metrics_name), index=False, float_format="%.3f")
        df.to_csv(dir_to_save / Path(metrics_name), index=False, float_format="%.3f")
    
    else:
        
        # Get the full metrics
        full_metrics = lightning_model.test_metrics
        
        # Convert tensors to float in the saved metrics
        for row in full_metrics:
            for key, value in row.items():
                if isinstance(value, torch.Tensor):
                    row[key] = value.item()
        
        # Get per-sample names for metrics
        if hasattr(dataset, "get_sample_names"):
            images_names = dataset.get_sample_names()
        else:
            images_names = dataset.get_images_names()

        # Get the original image names used to compare models across windows
        if hasattr(dataset, "get_original_image_names"):
            original_images_names = dataset.get_original_image_names()
        else:
            original_images_names = images_names
        
        ### --- Build the dataframe with one row per image and a mean row at the end
        
        # Create dataframe
        df = pd.DataFrame(full_metrics)
        
        # Add the mean row
        df.loc["mean"] = df.mean()
        
        # Add both the window name and the original image name
        df["image"] = images_names + ["mean"]
        df["original_image"] = original_images_names + ["mean"]
        df.set_index("image", inplace=True)
        
        # Save the metrics
        df.to_csv(training_path / Path(metrics_name), index=True, float_format="%.3f")
        df.to_csv(dir_to_save / Path(metrics_name), index=True, float_format="%.3f")
    
        
    # Save predictions if available
    if hasattr(lightning_model, "predictions"):
        
        # Get the predictions
        predictions = lightning_model.predictions
        
        
        # Create a folder to save the predictions
        predictions_path = training_path / Path(f"predictions_{dataset_name}")
        predictions_path.mkdir(exist_ok=True)
        
        # Get per-sample folder names for saving
        images_names = dataset.get_images_names()
        
        # Get the lr images
        lr_images = getattr(lightning_model, "inputs", None)
        prediction_meta = getattr(lightning_model, "prediction_meta", None)
        
        # Create a folder to save the lr images
        lr_path = predictions_path / Path("lr_images")
        lr_path.mkdir(exist_ok=True)
        
        def _to_uint8_image(tensor):
            img = tensor.detach().cpu().numpy()
            if img.ndim == 3:
                img = img.transpose(1, 2, 0)
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)
            return img.squeeze()

        # Check if predictions are images or tensors and save them
        for idx, denoised_image in enumerate(predictions):
            if not isinstance(denoised_image, torch.Tensor):
                continue

            meta = prediction_meta[idx] if prediction_meta and idx < len(prediction_meta) else {}
            folder = meta.get("folder")
            start = meta.get("start")
            lq_paths = meta.get("lq_paths")

            if folder is None:
                folder = images_names[idx] if idx < len(images_names) else f"sample_{idx:06d}"

            seq_base_dir = predictions_path / Path(folder)
            lr_base_dir = lr_path / Path(folder)
            seq_base_dir.mkdir(parents=True, exist_ok=True)
            lr_base_dir.mkdir(parents=True, exist_ok=True)

            if denoised_image.ndim == 4:
                t_out = denoised_image.shape[0]
                if t_out == 1 and lq_paths:
                    center_idx = len(lq_paths) // 2
                    frame_name = Path(lq_paths[center_idx]).name
                    io.imsave(seq_base_dir / Path(frame_name), _to_uint8_image(denoised_image[0]))
                    if lr_images is not None and idx < len(lr_images):
                        noisy = lr_images[idx]
                        if isinstance(noisy, torch.Tensor):
                            if noisy.ndim == 4 and center_idx < noisy.shape[0]:
                                io.imsave(lr_base_dir / Path(frame_name), _to_uint8_image(noisy[center_idx]))
                            elif noisy.ndim == 3:
                                io.imsave(lr_base_dir / Path(frame_name), _to_uint8_image(noisy))
                else:
                    seq_dir = seq_base_dir
                    if start is not None:
                        seq_dir = seq_dir / Path(f"start_{int(start):06d}")
                    seq_dir.mkdir(parents=True, exist_ok=True)

                    lr_seq_dir = lr_base_dir
                    if start is not None:
                        lr_seq_dir = lr_seq_dir / Path(f"start_{int(start):06d}")
                    lr_seq_dir.mkdir(parents=True, exist_ok=True)

                    for t in range(denoised_image.shape[0]):
                        frame_name = f"{t:06d}.png"
                        if lq_paths and t < len(lq_paths):
                            frame_name = Path(lq_paths[t]).name
                        io.imsave(seq_dir / Path(frame_name), _to_uint8_image(denoised_image[t]))
                        if lr_images is not None and idx < len(lr_images):
                            noisy = lr_images[idx]
                            if isinstance(noisy, torch.Tensor) and noisy.ndim == 4 and t < noisy.shape[0]:
                                io.imsave(lr_seq_dir / Path(frame_name), _to_uint8_image(noisy[t]))
            else:
                frame_name = "pred.png"
                if lq_paths:
                    center = len(lq_paths) // 2
                    frame_name = Path(lq_paths[center]).name
                io.imsave(seq_base_dir / Path(frame_name), _to_uint8_image(denoised_image))
                if lr_images is not None and idx < len(lr_images):
                    noisy = lr_images[idx]
                    if isinstance(noisy, torch.Tensor):
                        io.imsave(lr_base_dir / Path(frame_name), _to_uint8_image(noisy))

        
        
    log.info("Test finished!")
        



    