import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchviz import make_dot

from ptflops import get_model_complexity_info
from torchsummary import summary

import logging
import torch.profiler as profiler


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


def debug(cfg):
    log.info("Debugging...")


    ### =================== ###
    ### --- MODEL SETUP --- ###
    ### =================== ###

    # Load model
    model = hy_utils.instantiate(cfg.model)
    
    device = torch.device("cuda" if cfg.trainer.accelerator == "gpu" else "cpu")
    model = model.to(device)
    model.eval()
    

    
    ### ===================== ###
    ### --- DATASET SETUP --- ###
    ### ===================== ###

    # Load validation dataset
    dataset_val = hy_utils.instantiate(cfg.dataset, **cfg.dataset.val)
    

    ### ================= ###
    ### --- DEBUGGING --- ###
    ### ================= ###
    
    
    # First element of the dataset
    # x, y = dataset_val[0]
    
    # Get the model complexity
    macs, params = get_model_complexity_info(model, input_res=(3, 64, 64), as_strings=True, backend='aten', print_per_layer_stat=True, verbose=False)

    # Get the MACs and FLOPs
    macs_number, macs_unit = macs.split(" ")
    macs_number = float(macs_number)
    op_unit = macs_unit.removesuffix("Mac")
    flops = macs_number*2
    
    # Get the number of parameters
    params_number, params_unit = params.split(" ")
    params_number = float(params_number)
    
    
    # Create pandas dataframe
    df = pd.DataFrame({
        "Model": [cfg.model._target_.split(".")[1]],
        f"MACs ({op_unit}MAC)": [macs_number],
        f"FLOPs ({op_unit}FLOPs)": [flops],
        f"Params ({params_unit})": [params_number]
    })
    
    log.info("")
    log.info(f"\n=== MODEL COMPLEXITY ===\n{df}\n=======================")
    
    # Print parameters summary by layer/module
    # summary(model=model, input_size=(3, 64, 64))
    
    # Show graph
    # y_hat = model(torch.rand(1, 3, 64, 64).to(device))
    # make_dot(y_hat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

    # Profiler
    dummy_input = torch.rand(1, 3, 64, 64).to(device)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("model_inference"):
            output = model(dummy_input)

    # Print a summary of the most time‑consuming operations:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    
    print(torch.cuda.memory_summary())    
    
    log.info("Debugging finished!")
        



    