from copy import deepcopy
from typing import Any, cast

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
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
from lightning.pytorch.callbacks import TQDMProgressBar


log = logging.getLogger(__name__)

# Hydra config
hydra_config = HydraConfig.get()


class ResumeTQDMProgressBar(TQDMProgressBar):
    def on_train_start(self, *_) -> None:
        super().on_train_start(*_)

        bar = self.train_progress_bar

        if bar.total is None:
            bar.reset(total=self.total_train_batches)

def _prepare_resume_checkpoint(check_path: Path, reset_model_checkpoint_state: bool) -> Path:
    """Prepare checkpoint used for trainer resume.

    When starting a new Hydra run from an older checkpoint, restoring ModelCheckpoint
    callback state keeps the previous run's best path/score. This helper optionally
    strips ModelCheckpoint callback states while keeping model/optimizer/scheduler.
    """
    if not reset_model_checkpoint_state:
        return check_path

    checkpoint = torch.load(check_path, map_location="cpu", weights_only=False)
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        save_top_k=5,
        monitor="val/psnr",
        save_last=True,
        mode="max",
        filename="{step:08d}-{val_psnr:.4f}",
    )
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
    
    # Resolve checkpoint and loading mode.
    checkpoint_load_mode = cfg.get("checkpoint_load_mode", None)
    checkpoint_path = cfg.get("checkpoint", None)

    if checkpoint_load_mode not in (None, "resume", "finetune"):
        raise ValueError(
            f"Unknown checkpoint mode: {checkpoint_load_mode}"
        )

    if checkpoint_load_mode is None:
        if checkpoint_path is not None:
            raise ValueError(
                "A checkpoint was provided without checkpoint_load_mode."
            )
        check_path = None

    else:
        if checkpoint_path is None:
            raise ValueError(
                f"checkpoint_load_mode='{checkpoint_load_mode}' requires a checkpoint."
            )

        check_path = Path(checkpoint_path)

        # Allow a directory or a direct .ckpt path.
        if check_path.is_dir():
            candidates = sorted(check_path.rglob("last.ckpt"))

            if not candidates:
                raise FileNotFoundError(
                    f"No last.ckpt found in {check_path}"
                )

            if len(candidates) > 1:
                raise ValueError(
                    f"Multiple last.ckpt files found in {check_path}. "
                    "Specify the checkpoint file explicitly."
                )

            check_path = candidates[0]

        if not check_path.is_file() or check_path.suffix != ".ckpt":
            raise FileNotFoundError(
                f"Invalid checkpoint: {check_path}"
            )

        log.info(
            f"Checkpoint mode: {checkpoint_load_mode}, path: {check_path}"
        )


    
    # Instantiate LightningModule.
    lightning_model = hy_utils.instantiate(
        cfg.litmodule,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=scheduler,
    )

    fit_ckpt_path = None

    if checkpoint_load_mode == "resume":
        assert check_path is not None

        fit_ckpt_path = _prepare_resume_checkpoint(
            check_path,
            reset_model_checkpoint_state=cfg.get(
                "reset_best_on_resume", False
            ),
        )

        log.info(f"Resuming training from {fit_ckpt_path}")

    elif checkpoint_load_mode == "finetune":
        assert check_path is not None

        checkpoint = torch.load(
            check_path,
            map_location="cpu",
            weights_only=False,
        )

        lightning_model.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        log.info(f"Fine-tuning from {check_path}")

    else:
        log.info("Training from scratch")




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
        ResumeTQDMProgressBar(),
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

    print("Dataset samples:", len(dataset))
    print("Dataloader batches:", len(train_dataloader))
    print("Max epochs:", trainer.max_epochs)
    print("Max steps:", trainer.max_steps)

    # Train the model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=fit_ckpt_path,
    )    
    trainer.save_checkpoint(checkpoints_path / Path("stopped.ckpt"))    
    log.info("Training finished!")
    
    
    # Get the best model path
    best_model_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path != "" else on_exception_checkpoint_callback.ckpt_path

    
    # Duplicate the best model to a new file.
    if not best_model_path:
        raise RuntimeError("No best checkpoint was produced.")

    best_model_path = Path(best_model_path)

    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_model_path}"
        )

    new_best_model_path = checkpoints_path / "best.ckpt"

    if best_model_path.resolve() != new_best_model_path.resolve():
        shutil.copy2(best_model_path, new_best_model_path)
    
    # Print the best model path
    log.info(f"Best model saved at {new_best_model_path}")
    



    