from pathlib import Path
from copy import deepcopy

import torch
import lightning.pytorch as pl
from torch.optim.swa_utils import AveragedModel


def get_ema_avg_fn_compat(decay: float):
    """
    EMA averaging function compatible with older PyTorch AveragedModel versions.

    EMA update:
        ema = decay * ema + (1 - decay) * current
    """

    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1.0 - decay) * current_param

    return ema_update


class EMACheckpointCallback(pl.Callback):
    """
    Maintains an EMA version of the LightningModule and saves EMA checkpoints.

    Behavior:
      - Training uses normal model weights.
      - EMA weights are updated once per optimizer step.
      - latest_ema.ckpt is saved after validation.
      - last_ema.ckpt is saved at the end of training.
      - Normal model weights are restored immediately after saving EMA.
    """

    def __init__(
        self,
        decay: float = 0.999,
        start_step: int = 0,
        save_last_ema: bool = True,
        ema_ckpt_name: str = "last_ema.ckpt",
        use_buffers: bool = True,
        save_on_validation: bool = True,
        latest_ema_ckpt_name: str = "latest_ema.ckpt",
    ):
        super().__init__()

        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.start_step = start_step
        self.save_last_ema = save_last_ema
        self.ema_ckpt_name = ema_ckpt_name
        self.use_buffers = use_buffers
        self.save_on_validation = save_on_validation
        self.latest_ema_ckpt_name = latest_ema_ckpt_name

        self.ema_model = None
        self._loaded_ema_state = None

    def on_fit_start(self, trainer, pl_module):
        """
        Create EMA model after Lightning has moved the module to the right device.
        """
        self.ema_model = AveragedModel(
            pl_module,
            avg_fn=get_ema_avg_fn_compat(self.decay),
            use_buffers=self.use_buffers,
        )

        if self._loaded_ema_state is not None:
            self.ema_model.load_state_dict(self._loaded_ema_state)
            self._loaded_ema_state = None

    @torch.no_grad()
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """
        Update EMA once per optimizer step.

        This is better than on_train_batch_end when using gradient accumulation,
        because on_train_batch_end can run more often than optimizer.step().
        """
        if trainer.global_step < self.start_step:
            return

        if self.ema_model is None:
            return

        self.ema_model.update_parameters(pl_module)

    def state_dict(self):
        """
        Save EMA state inside Lightning checkpoints, so resuming training also
        resumes EMA.
        """
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "save_last_ema": self.save_last_ema,
            "ema_ckpt_name": self.ema_ckpt_name,
            "use_buffers": self.use_buffers,
            "save_on_validation": self.save_on_validation,
            "latest_ema_ckpt_name": self.latest_ema_ckpt_name,
            "ema_model": self.ema_model.state_dict()
            if self.ema_model is not None
            else None,
        }

    def load_state_dict(self, state_dict):
        """
        Restore EMA callback state when resuming from a checkpoint.
        """
        self.decay = state_dict.get("decay", self.decay)
        self.start_step = state_dict.get("start_step", self.start_step)
        self.save_last_ema = state_dict.get("save_last_ema", self.save_last_ema)
        self.ema_ckpt_name = state_dict.get("ema_ckpt_name", self.ema_ckpt_name)
        self.use_buffers = state_dict.get("use_buffers", self.use_buffers)
        self.save_on_validation = state_dict.get(
            "save_on_validation",
            self.save_on_validation,
        )
        self.latest_ema_ckpt_name = state_dict.get(
            "latest_ema_ckpt_name",
            self.latest_ema_ckpt_name,
        )

        self._loaded_ema_state = state_dict.get("ema_model", None)

    def _get_ckpt_dir(self, trainer) -> Path:
        ckpt_callback = trainer.checkpoint_callback

        if ckpt_callback is not None and ckpt_callback.dirpath is not None:
            return Path(ckpt_callback.dirpath)

        return Path(trainer.default_root_dir) / "checkpoints"

    @torch.no_grad()
    def save_ema_checkpoint(self, trainer, pl_module, path):
        """
        Temporarily replace pl_module weights with EMA weights, save checkpoint,
        then restore the original training weights.
        """
        if self.ema_model is None:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        original_state = deepcopy(pl_module.state_dict())
        ema_state = self.ema_model.module.state_dict()

        try:
            pl_module.load_state_dict(ema_state, strict=True)
            trainer.save_checkpoint(str(path))
        finally:
            pl_module.load_state_dict(original_state, strict=True)

    def on_validation_end(self, trainer, pl_module):
        """
        Save latest EMA checkpoint after each validation.
        This overwrites the previous latest_ema.ckpt.
        """
        if not self.save_on_validation:
            return

        if self.ema_model is None:
            return

        if trainer.global_step < self.start_step:
            return

        ckpt_dir = self._get_ckpt_dir(trainer)
        ema_path = ckpt_dir / self.latest_ema_ckpt_name

        self.save_ema_checkpoint(trainer, pl_module, ema_path)

        # print(f"[EMA] Saved latest EMA checkpoint to: {ema_path}")

    def on_train_end(self, trainer, pl_module):
        """
        Save final EMA checkpoint at the end of training.
        """
        if not self.save_last_ema:
            return

        if self.ema_model is None:
            return

        ckpt_dir = self._get_ckpt_dir(trainer)
        ema_path = ckpt_dir / self.ema_ckpt_name

        self.save_ema_checkpoint(trainer, pl_module, ema_path)

        # print(f"[EMA] Saved final EMA checkpoint to: {ema_path}")