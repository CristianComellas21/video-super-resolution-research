from copy import deepcopy
from typing import Any, cast

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from torch import nn
import torch
import tempfile
import shutil

import logging



# Hydra and lightning imports
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch import loggers as pl_loggers
from hydra.core.hydra_config import HydraConfig
import hydra.utils as hy_utils

from callbacks.ema_checkpoint import EMACheckpointCallback

from pathlib import Path


log = logging.getLogger(__name__)

# Hydra config
hydra_config = HydraConfig.get()


def _prepare_resume_checkpoint(check_path: Path, reset_model_checkpoint_state: bool) -> Path:
    """Prepare checkpoint used for trainer resume.

    When starting a new Hydra run from an older checkpoint, restoring ModelCheckpoint
    callback state keeps the previous run's best path/score. This helper optionally
    strips ModelCheckpoint callback states while keeping model/optimizer/scheduler.
    """
    if not reset_model_checkpoint_state:
        return check_path

    checkpoint = torch.load(check_path, map_location="cpu")
    callbacks = checkpoint.get("callbacks", None)
    if not isinstance(callbacks, dict):
        return check_path

    filtered_callbacks = {
        key: value
        for key, value in callbacks.items()
        if "ModelCheckpoint" not in key
    }
    if len(filtered_callbacks) == len(callbacks):
        return check_path

    checkpoint["callbacks"] = filtered_callbacks
    tmp_dir = Path(tempfile.mkdtemp(prefix="resume_ckpt_"))
    tmp_path = tmp_dir / check_path.name
    torch.save(checkpoint, tmp_path)
    return tmp_path


def _extract_epoch_from_checkpoint(path: Path, on_exception_filename: str) -> int:
    stem = path.stem
    if stem == on_exception_filename:
        return 0

    if "epoch=" in stem:
        token = stem.split("epoch=")[1].split("-")[0]
    else:
        token = stem.split("-")[0]

    return int(token)

def train(cfg):
    log.info("Training...")
    
    ### =================== ###
    ### --- MODEL SETUP --- ###
    ### =================== ###

    # Load model
    model = hy_utils.instantiate(cfg.model)

    ### ======================= ###
    ### --- OPTIMIZER SETUP --- ###
    ### ======================= ###

    # Load optimizer
    optimizer_target = cfg.optimizer.get("_target_", "")
    
    params = filter(lambda p: p.requires_grad, model.parameters())

    if "DeepSpeedCPUAdam" in optimizer_target:
        optimizer = hy_utils.instantiate(cfg.optimizer, model_params=params)
    else:
        optimizer = hy_utils.instantiate(cfg.optimizer, params=params)
    
    # Load scheduler
    scheduler = hy_utils.instantiate(cfg.scheduler, optimizer=optimizer)


    ### ======================= ###
    ### --- CRITERION SETUP --- ###
    ### ======================= ###
    
    # Load criterion
    criterion = hy_utils.instantiate(cfg.criterion)



    ### ===================== ###
    ### --- DATASET SETUP --- ###
    ### ===================== ###

    # Load main dataset

    gt_root = cfg.dataset.get("dataroot_gt", None)
    lq_root = cfg.dataset.get("dataroot_lq", None)
    if lq_root is not None:
        log.info(f"Loading dataset from {gt_root} and {lq_root}")
    else:
        log.info(f"Loading dataset from {gt_root}")
    
    dataset = hy_utils.instantiate(cfg.dataset, **cfg.dataset.train)
    
    # Load validation dataset
    dataset_val = hy_utils.instantiate(cfg.dataset, **cfg.dataset.val)
    
    
    ### ======================== ###
    ### --- DATALOADER SETUP --- ###
    ### ======================== ###

    sampler = {"data_source": dataset} if cfg.dataloader.train.get("sampler", None) is not None else None
    train_dataloader = hy_utils.instantiate(cfg.dataloader.train, dataset=dataset, sampler=sampler)
    
    # Load validation dataloader
    val_dataloader = hy_utils.instantiate(cfg.dataloader.val, dataset=dataset_val)


    ### ====================== ###
    ### --- TRAINING SETUP --- ###
    ### ====================== ###

    # --- Define logger and checkpoint parameters for the trainer

    # Info from the hydra config
    dir_to_save = hydra_config.runtime.output_dir
    job_name = hydra_config.job.name

    # Checkpoint callbacks
    checkpoints_path = dir_to_save / Path("checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, save_top_k=5, monitor="val/psnr", save_last=True, mode="max", filename="{epoch:05d}-{val_psnr:.4f}",)

    on_exception_filename = "on_exception"
    on_exception_checkpoint_callback = OnExceptionCheckpoint(dirpath=checkpoints_path, filename=on_exception_filename)

    # Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_to_save, name="", version="")
    
    # Redirect the lightning logs to the hydra logs
    logger = logging.getLogger("lightning.pytorch")
    logger.addHandler(logging.FileHandler(dir_to_save / Path(f"{job_name}.log")))


    ### ===================== ###
    ### --- TRAINING LOOP --- ###
    ### ===================== ###


    # Load pretrained model path if it exists, otherwise train from scratch
    check_path = cfg.get("checkpoint", None)
    
    if check_path is not None:
        check_path = Path(check_path)
        log.info(f"Loading model from {check_path}")
        
        if not check_path.exists():
            log.info(f"Checkpoint path {check_path} does not exist. Training from scratch.")
            check_path = None
        elif check_path.is_dir():
            try:
                check_path = next(check_path.rglob("last*.ckpt"))
            except StopIteration:
                try:
                    check_path = next(check_path.rglob("best*.ckpt"))
                except StopIteration:
                    check_path = None
                    log.info(f"Checkpoint path {check_path} is a directory and does not contain the required .ckpt file. Training from scratch.")
                
        elif check_path.suffix != ".ckpt":
            log.info(f"Checkpoint path {check_path} is not a .ckpt file. Training from scratch.")
            check_path = None

    resume_check_path = None
    if check_path is not None:
        reset_best_on_resume = cfg.get("reset_best_on_resume", True)
        resume_check_path = _prepare_resume_checkpoint(
            check_path,
            reset_model_checkpoint_state=reset_best_on_resume,
        )
        if resume_check_path != check_path:
            log.info("ModelCheckpoint state reset for resumed run (best metric/path starts fresh).")
    

    # # TODO: remove this patch
    # # Change some keys of the state_dict to match the new model
    # # I have to change *L2AttentionHead* to *attention*
    # check = torch.load(check_path)
    # state_dict = check["state_dict"]
    # state_dict = {k.replace("L2AttentionHead", "attention") if "L2AttentionHead" in k else k: v for k, v in state_dict.items()}
    # check["state_dict"] = state_dict
    # tmp_file_path = checkpoints_path / Path("tmp.ckpt")
    # torch.save(check, tmp_file_path)
    # # model.load_state_dict(state_dict)

    
    ### ======================= ###
    ### --- LITMODULE SETUP --- ###
    ### ======================= ###


    if check_path is None:    
        # Load the model from scratch
        log.info("Training from scratch")
        lightning_model = hy_utils.instantiate(cfg.litmodule, model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=scheduler)
    
    else: 
        if cfg.override_hyperparams:
            
            # Load the model from the checkpoint and override the hyperparameters to resume training
            lightning_model = hy_utils.get_class(cfg.litmodule._target_).load_from_checkpoint(check_path, map_location="cpu", model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=scheduler, override_hyperparams=True)

        else:
            # Load the model from the checkpoint and keep the hyperparameters from the checkpoint
            log.info(f"Loading model from {check_path}")
            # lightning_model = hy_utils.get_class(cfg.litmodule._target_).load_from_checkpoint(check_path)    
            lightning_model = hy_utils.get_class(cfg.litmodule._target_).load_from_checkpoint(check_path, map_location="cpu", model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=scheduler)    
    


    # Define trainer
    trainer_cfg = cast(
        dict[str, Any],
        OmegaConf.to_container(
            cfg.trainer,
            resolve=True,
            throw_on_missing=True,
        ),
    )

    if isinstance(cfg.trainer.get("strategy"), DictConfig):
        strategy_cfg = cast(
            dict[str, Any],
            OmegaConf.to_container(
                cfg.trainer.strategy,
                resolve=True,
                throw_on_missing=True,
            ),
        )

        strategy_target = strategy_cfg.get("_target_", "")

        if "FSDPStrategy" in strategy_target:
            ignored_names = strategy_cfg.pop("ignored_modules", [])

            ignored_modules = []
            for name in ignored_names:
                module = lightning_model
                for attr in name.split("."):
                    module = getattr(module, attr)
                ignored_modules.append(module)

            trainer_cfg["strategy"] = hy_utils.instantiate(
                strategy_cfg,
                ignored_modules=ignored_modules,
            )

        else:
            # DeepSpeed / other strategies
            trainer_cfg["strategy"] = hy_utils.instantiate(strategy_cfg)



    ema_callback = None

    if cfg.get("ema", None) is not None and cfg.ema.get("enabled", False):
        ema_callback = EMACheckpointCallback(
            decay=cfg.ema.get("decay", 0.999),
            start_step=cfg.ema.get("start_step", 0),
            save_last_ema=cfg.ema.get("save_last_ema", True),
            ema_ckpt_name=cfg.ema.get("ema_ckpt_name", "last_ema.ckpt"),
            use_buffers=cfg.ema.get("use_buffers", True),
            save_on_validation=cfg.ema.get("save_on_validation", True),
            latest_ema_ckpt_name=cfg.ema.get("latest_ema_ckpt_name", "latest_ema.ckpt"),
        )

    callbacks = [
        checkpoint_callback,
        on_exception_checkpoint_callback,
    ]

    if ema_callback is not None:
        callbacks.append(ema_callback)

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **trainer_cfg,
    )


    # trainer = pl.Trainer(
    #     logger=tb_logger,
    #     # callbacks=[checkpoint_callback, on_exception_checkpoint_callback],
    #     callbacks=[checkpoint_callback],
    #     **trainer_cfg,
    # )



    # Train the model
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=resume_check_path)
    trainer.save_checkpoint(checkpoints_path / Path("stopped.ckpt"))
    
    log.info("Training finished!")
    
    
    # Get the best model path
    best_model_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path != "" else on_exception_checkpoint_callback.ckpt_path

    # Get epoch of the best model
    best_model_path = Path(best_model_path)
    best_model_epoch = _extract_epoch_from_checkpoint(best_model_path, on_exception_filename)
    

    # Duplicate the best model to a new file
    best_model_path = Path(best_model_path)
    new_best_model_path = checkpoints_path / Path(f"best-epoch={best_model_epoch}.ckpt")
    shutil.copy2(best_model_path, new_best_model_path)
    
    # Print the best model path
    log.info(f"Best model saved at {new_best_model_path}")
    



    