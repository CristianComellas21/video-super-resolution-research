import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure


class LitModule(pl.LightningModule):

    def __init__(self, model: nn.Module, optimizer=None, criterion=F.mse_loss, lr_scheduler=None, **kwargs):
        super().__init__()
        
        # Hyperparameters
        self.save_hyperparameters()
                
        # Precision
        torch.set_float32_matmul_precision('medium')
        
        # Set attributes
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
                


    #### CONFIGURATION FUNCTIONS ####

    def setup(self, stage: str) -> None:
        """ Setup the model and move it to the device. """

        self.model = self.model.to(self.device)
        

        # Setup variables to save the test metrics of each test step
        # Also setup variable to save the predictions (in order to avoid recomputing them with the predict_step method)
        if stage == "test":
            self.test_metrics = []
            self.store_predictions = getattr(self.hparams, "store_predictions", True)
            if self.store_predictions:
                self.predictions = []
                self.inputs = []
            self.prediction_meta = []

        


        return super().setup(stage)
    

    def set_properties_model(self, properties: dict):
        """Set properties to the model."""
        for key, value in properties.items():
            setattr(self.model, key, value)


    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': 'average_val_loss'
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if getattr(self.hparams, "override_hyperparams", False):
            # Strip the optimizer and scheduler states to override hyperparameters.
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []


    def on_train_epoch_end(self):
        # Check if the dataset has a method to set the use_cache attribute
        if not hasattr(self.trainer.train_dataloader.dataset, "set_use_cache"):
            return
        
        if not self.trainer.train_dataloader.dataset.get_use_cache():
            self.trainer.train_dataloader.dataset.set_use_cache(True)
    
    
    def on_validation_epoch_end(self):        
        # Check if the dataset has a method to set the use_cache attribute
        if not hasattr(self.trainer.val_dataloaders.dataset, "set_use_cache"):
            return
        
        if not self.trainer.val_dataloaders.dataset.get_use_cache():
            self.trainer.val_dataloaders.dataset.set_use_cache(True)


    #### STEP FUNCTIONS ####

    def training_step(self, batch, batch_idx):


        # Check if an optimizer was provided
        if self.optimizer is None:
            raise ValueError(f"Optimizer is None. Please provide an optimizer to the lightning module to train the model.")
        
        
        # Get data from the batch
        x, y = self.__get_xy(batch)

        # Forward pass
        y_hat = self.model(x)
        y_hat, y = self._align_temporal(y_hat, y)

        # Calculate loss
        out = self.criterion(y_hat, y)
        loss, *extras = out if isinstance(out, (tuple, list)) else (out,)
        
        batch_size = self._get_batch_size(x)

        # Log loss
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        
        self._log_loss_extras(
            prefix="train_loss",
            extras=extras,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
        )
        
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(
            "lr",
            lr,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

        return loss



    def validation_step(self, batch, batch_idx):

        # Get data from the batch
        x, y = self.__get_xy(batch)

        # Forward pass
        y_hat = self.model(x)
        y_hat, y = self._align_temporal(y_hat, y)

        # Calculate loss
        out = self.criterion(y_hat, y)
        val_loss, *extras = out if isinstance(out, (tuple, list)) else (out,)
        
        batch_size = self._get_batch_size(x)

        # Log loss
        self.log("average_val_loss", val_loss, batch_size=batch_size)
        self._log_loss_extras(
            prefix="val_loss",
            extras=extras,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )


    def test_step(self, batch, batch_idx):
        # Get data from the batch
        x, y = self.__get_xy(batch)

        y_hat = self._maybe_forward_tiled(x)
        y_hat, y = self._align_temporal(y_hat, y)
        x_for_save = self._align_input_to_output(x, y_hat)
        
        ######################################################################################
        #### DEBUG LAPLACIAN ENERGY TO CHECK HIGH-FREQUENCY CONTENT OF PREDICTIONS AND GT ####
        ######################################################################################
        
        # def lap_energy(x):
        #     # x: (N,T,C,H,W) o (N,C,H,W)
        #     if x.ndim == 5:
        #         x = x.flatten(0,1)
        #     k = x.new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]]).view(1,1,3,3)
        #     k = k.repeat(x.size(1),1,1,1)
        #     y = F.conv2d(x, k, padding=1, groups=x.size(1))
        #     return y.abs().mean()

        # base = F.interpolate(x[:, x.shape[1]//2], scale_factor=4, mode="bicubic", align_corners=False)
        # pred = y_hat[:,0]  # si es (N,1,C,H,W)
        # res  = pred - base
        
        # print("[litmodule.test_step] lap(base):", lap_energy(base).item())
        # print("[litmodule.test_step] lap(pred):", lap_energy(pred).item())
        # print("[litmodule.test_step] lap(res): ", lap_energy(res).item())
        # print("[litmodule.test_step] lap(gt):  ", lap_energy(y).item())
        
        # print(f"[litmodule.test_step] y_hat range: {y_hat.min().item()} to {y_hat.max().item()}")
        # print(f"[litmodule.test_step] gt range: {y.min().item()} to {y.max().item()}")
        
        ######################################################################################
        ######################################################################################
        ######################################################################################
        
        # Ensure y_hat and y have the same shape
        y = self._crop_spatial(y, crop=0)
        y_hat = self._crop_spatial(y_hat, crop=0)
        
        out = self.criterion(y_hat, y)
        test_loss, *extras = out if isinstance(out, (tuple, list)) else (out,)
        psnr = peak_signal_noise_ratio(*self._flatten_video_for_metrics(y_hat, y), data_range=1.0)
        ssim = structural_similarity_index_measure(*self._flatten_video_for_metrics(y_hat, y), data_range=1.0)

        metrics = {
            "test_loss": test_loss,
            "psnr": psnr,
            "ssim": ssim
        }
        
        if extras:
            self._log_loss_extras(
                prefix="test_loss",
                extras=extras,
                batch_size=self._get_batch_size(x),
                on_step=False,
                on_epoch=True,
            )

        self.test_metrics.append(metrics)
        if getattr(self, "store_predictions", True):
            self.predictions.extend([y_hat[i, ...] for i in range(y_hat.shape[0])])
            self.inputs.extend([x_for_save[i, ...] for i in range(x_for_save.shape[0])])

        if isinstance(batch, dict) and "lq_path" in batch:
            batch_size = y_hat.shape[0] if y_hat.ndim >= 1 else 1
            t_out = y_hat.shape[1] if y_hat.ndim == 5 else 1

            folders = self._normalize_batch_field(batch.get("folder"), batch_size)
            starts = self._normalize_batch_field(batch.get("start"), batch_size)
            lq_paths_per_sample = self._normalize_lq_paths(batch.get("lq_path"), batch_size)

            temporal_offset = 0
            if x.ndim == 5 and y_hat.ndim == 5 and x.shape[1] >= y_hat.shape[1]:
                temporal_offset = (x.shape[1] - y_hat.shape[1]) // 2

            for i in range(batch_size):
                paths = self._center_crop_sequence(lq_paths_per_sample[i], t_out)

                meta = {"lq_paths": paths}

                folder_i = folders[i] if i < len(folders) else None
                if folder_i is not None:
                    meta["folder"] = folder_i

                start_i = starts[i] if i < len(starts) else None
                if isinstance(start_i, torch.Tensor) and start_i.numel() == 1:
                    start_i = int(start_i.item())
                elif isinstance(start_i, torch.Tensor):
                    start_i = None

                if isinstance(start_i, int):
                    meta["start"] = start_i + temporal_offset
                elif start_i is not None:
                    meta["start"] = start_i

                self.prediction_meta.append(meta)

        batch_size = self._get_batch_size(x)
        self.log_dict(metrics, batch_size=batch_size)

        return metrics
        
    def predict_step(self, batch, batch_idx):
        # Get data from the batch
        x, _ = self.__get_xy(batch)

        return self._maybe_forward_tiled(x)


    ### Auxiliary functions ###

    # -- Preparation functions -- #

    def __get_xy(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Process a batch of data and returns x and y tensors.

            Args:
                batch: A batch of data.
                
        """
        if isinstance(batch, dict):
            x = batch.get("L")
            y = batch.get("H")
            return x, y

        x, y, _ = batch
        return x, y

    def _crop_spatial(self, tensor, crop=10):
        if crop <= 0:
            return tensor
        return tensor[..., crop:-crop, crop:-crop]

    def _flatten_video_for_metrics(self, y_hat, y):
        y_hat, y = self._align_temporal(y_hat, y)
        b, t, c, h, w = y_hat.shape
        y_hat = y_hat.reshape(b * t, c, h, w)
        y = y.reshape(b * t, c, h, w)
        return y_hat, y

    def _align_input_to_output(self, x, y_hat):
        if not (isinstance(x, torch.Tensor) and isinstance(y_hat, torch.Tensor)):
            return x
        if x.ndim != 5 or y_hat.ndim != 5:
            return x

        t_in = x.shape[1]
        t_out = y_hat.shape[1]
        if t_in == t_out:
            return x
        if t_in > t_out:
            start = (t_in - t_out) // 2
            return x[:, start:start + t_out]
        return x

    def _align_temporal(self, y_hat, y):
        if y_hat.ndim != 5 or y.ndim != 5:
            return y_hat, y
        t_hat = y_hat.shape[1]
        t = y.shape[1]
        if t_hat == t:
            return y_hat, y
        if t_hat < t:
            # Center-aligned slice from GT to match prediction length.
            start = (t - t_hat) // 2
            return y_hat, y[:, start:start + t_hat]
        # If prediction is longer, center-crop prediction to GT length.
        start = (t_hat - t) // 2
        return y_hat[:, start:start + t], y

    def _maybe_forward_tiled(self, x: torch.Tensor) -> torch.Tensor:
        tile_size = getattr(self.hparams, "test_tile_size", None)
        if tile_size is None:
            return self.model(x)
        overlap = getattr(self.hparams, "test_tile_overlap", 0.25)
        weight_type = getattr(self.hparams, "test_tile_weight", "hann")
        return self._forward_tiled(x, tile_size=tile_size, overlap=overlap, weight_type=weight_type)

    def _forward_tiled(
        self,
        x: torch.Tensor,
        tile_size,
        overlap: float = 0.25,
        weight_type: str = "hann",
    ) -> torch.Tensor:
        if x.ndim not in (4, 5):
            return self.model(x)

        h, w = x.shape[-2], x.shape[-1]
        if isinstance(tile_size, int):
            tile_h, tile_w = tile_size, tile_size
        else:
            tile_h, tile_w = tile_size

        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)

        overlap_h = self._normalize_overlap(overlap, tile_h)
        overlap_w = self._normalize_overlap(overlap, tile_w)

        ys = self._tile_positions(h, tile_h, overlap_h)
        xs = self._tile_positions(w, tile_w, overlap_w)

        # Probe first tile to infer output scale.
        probe = x[..., ys[0]:ys[0] + tile_h, xs[0]:xs[0] + tile_w]
        probe_out = self.model(probe)
        scale_h = probe_out.shape[-2] / probe.shape[-2]
        scale_w = probe_out.shape[-1] / probe.shape[-1]
        out_h = int(round(h * scale_h))
        out_w = int(round(w * scale_w))

        out_shape = list(probe_out.shape)
        out_shape[-2] = out_h
        out_shape[-1] = out_w
        output = torch.zeros(out_shape, device=probe_out.device, dtype=probe_out.dtype)
        weight_sum = torch.zeros_like(output)

        for y0 in ys:
            for x0 in xs:
                y1 = y0 + tile_h
                x1 = x0 + tile_w
                tile = x[..., y0:y1, x0:x1]
                tile_out = self.model(tile)

                out_y0 = int(round(y0 * scale_h))
                out_x0 = int(round(x0 * scale_w))
                out_y1 = out_y0 + tile_out.shape[-2]
                out_x1 = out_x0 + tile_out.shape[-1]

                weight = self._build_weight_window(
                    tile_out.shape[-2],
                    tile_out.shape[-1],
                    weight_type,
                    device=tile_out.device,
                    dtype=tile_out.dtype,
                    ndim=tile_out.ndim,
                )

                output[..., out_y0:out_y1, out_x0:out_x1] += tile_out * weight
                weight_sum[..., out_y0:out_y1, out_x0:out_x1] += weight

        output = output / torch.clamp(weight_sum, min=1e-8)
        return output


    def _log_loss_extras(
        self,
        prefix: str,
        extras,
        batch_size: int | None,
        on_step: bool,
        on_epoch: bool,
    ):
        """
        extras: resultado del unpacking *extras del criterion.
        Esperamos como máximo un dict.
        Si no cumple el formato, no loguea nada.
        """

        if not extras:
            return

        # Solo aceptamos exactamente un dict
        if len(extras) != 1 or not isinstance(extras[0], dict):
            return

        log_dict = extras[0]

        for name, value in log_dict.items():
            if torch.is_tensor(value):
                self.log(
                    f"{prefix}/{name}",
                    value.detach(),
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )


    @staticmethod
    def _normalize_overlap(overlap, tile_size: int) -> int:
        if isinstance(overlap, float):
            return int(round(tile_size * overlap))
        return int(overlap)

    @staticmethod
    def _normalize_batch_field(field, batch_size: int):
        if isinstance(field, torch.Tensor):
            if field.ndim == 0:
                return [field.item()] * batch_size
            return field.detach().cpu().tolist()

        if isinstance(field, (list, tuple)):
            values = list(field)
            if len(values) == batch_size:
                return values
            if len(values) == 1 and batch_size > 1:
                return values * batch_size
            if len(values) < batch_size:
                return values + [None] * (batch_size - len(values))
            return values[:batch_size]

        return [field] * batch_size

    @staticmethod
    def _normalize_lq_paths(lq_paths, batch_size: int):
        if lq_paths is None:
            return [[] for _ in range(batch_size)]

        # Default DataLoader collate for list[str] produces a transposed structure:
        # [time][batch] instead of [batch][time].
        if isinstance(lq_paths, (list, tuple)) and lq_paths:
            first = lq_paths[0]
            if isinstance(first, (list, tuple)):
                t = len(lq_paths)
                b = len(first)
                per_sample = []
                for bi in range(min(batch_size, b)):
                    per_sample.append([lq_paths[ti][bi] for ti in range(t)])
                if len(per_sample) < batch_size:
                    per_sample.extend([[] for _ in range(batch_size - len(per_sample))])
                return per_sample

            if len(lq_paths) == batch_size and all(isinstance(v, (list, tuple)) for v in lq_paths):
                return [list(v) for v in lq_paths]

            if len(lq_paths) == batch_size and all(isinstance(v, str) for v in lq_paths):
                return [[v] for v in lq_paths]

            if batch_size == 1 and all(isinstance(v, str) for v in lq_paths):
                return [list(lq_paths)]

        if isinstance(lq_paths, str):
            return [[lq_paths] for _ in range(batch_size)]

        return [[] for _ in range(batch_size)]

    @staticmethod
    def _center_crop_sequence(seq, target_len: int):
        if not isinstance(seq, (list, tuple)):
            return seq
        seq = list(seq)
        if target_len <= 0 or len(seq) <= target_len:
            return seq
        start = (len(seq) - target_len) // 2
        return seq[start:start + target_len]

    @staticmethod
    def _tile_positions(length: int, tile_size: int, overlap_px: int) -> list[int]:
        tile_size = min(tile_size, length)
        if tile_size >= length:
            return [0]
        step = tile_size - overlap_px
        if step <= 0:
            raise ValueError("Tile overlap too large; step would be <= 0.")
        positions = list(range(0, length - tile_size + 1, step))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    @staticmethod
    def _build_weight_window(
        height: int,
        width: int,
        weight_type: str,
        device,
        dtype,
        ndim: int,
    ) -> torch.Tensor:
        if weight_type == "hann":
            wy = torch.hann_window(height, device=device, dtype=dtype)
            wx = torch.hann_window(width, device=device, dtype=dtype)
            w2d = wy[:, None] * wx[None, :]
        else:
            w2d = torch.ones((height, width), device=device, dtype=dtype)

        if ndim == 5:
            return w2d[None, None, None, :, :]
        return w2d[None, None, :, :]

    @staticmethod
    def _get_batch_size(x):
        if isinstance(x, torch.Tensor) and x.ndim > 0:
            return x.shape[0]
        return None

