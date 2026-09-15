import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity


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

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True,
        )
        self.dists = DeepImageStructureAndTextureSimilarity()

                

    #### CONFIGURATION FUNCTIONS ####

    def setup(self, stage: str) -> None:
        """ Setup the model and move it to the device. """

        self.model = self.model.to(self.device)
        
        # Setup variables to save the test metrics of each test step
        # Also setup variable to save the predictions (in order to avoid recomputing them with the predict_step method)
        if stage == "test":
            self.test_metrics = []
            self.store_predictions = getattr(self.hparams, "store_predictions", True)
            self.diagnose_difficulty = getattr(self.hparams, "diagnose_difficulty", False)
            
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
        if self.optimizer is None:
            raise ValueError("Optimizer is None.")

        if self.lr_scheduler is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if getattr(self.hparams, "override_hyperparams", False):
            # Strip the optimizer and scheduler states to override hyperparameters.
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []

    def on_train_start(self) -> None:
        """Debug helper: print modules that are not in training mode and frozen params."""
        try:
            print("[LitAdvanced.on_train_start] model.training:", getattr(self.model, "training", None))
            # flownet commonly is set to eval when frozen; report it explicitly if present
            if hasattr(self.model, "flownet"):
                print("[LitAdvanced.on_train_start] model.flownet.training:", getattr(self.model.flownet, "training", None))

            not_training = []
            for name, m in self.model.named_modules():
                if not m.training:
                    not_training.append((name, type(m).__name__))

            if not_training:
                print("[LitAdvanced] Modules NOT in training mode:")
                for name, tname in not_training:
                    print(" -", name or "<root>", "(", tname, ")")
            else:
                print("[LitAdvanced] All modules are in training mode.")

            # List frozen parameters
            frozen = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    frozen.append(name)
            if frozen:
                print("[LitAdvanced] Frozen parameters (requires_grad=False):")
                for n in frozen:
                    print(" -", n)
            else:
                print("[LitAdvanced] No frozen parameters detected.")
        except Exception as e:
            print("[LitAdvanced.on_train_start] Debug printing failed:", e)


    def on_train_epoch_end(self):
        # Check if the dataset has a method to set the use_cache attribute
        if not hasattr(self.trainer.train_dataloader.dataset, "set_use_cache"):
            return
        
        if not self.trainer.train_dataloader.dataset.get_use_cache():
            self.trainer.train_dataloader.dataset.set_use_cache(True)
    
    
    def on_validation_epoch_end(self):        
        # Check if the dataset has a method to set the use_cache attribute
        if isinstance(self.trainer.val_dataloaders, (list, tuple)):
            if len(self.trainer.val_dataloaders) == 0:
                return
            val_dataset = self.trainer.val_dataloaders[0].dataset
        else:
            val_dataset = self.trainer.val_dataloaders.dataset

        if not hasattr(val_dataset, "set_use_cache"):
            return
        
        if not val_dataset.get_use_cache():
            val_dataset.set_use_cache(True)


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
        self.log(
            "average_val_loss",
            val_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        self._log_loss_extras(
            prefix="val_loss",
            extras=extras,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        # Prepare tensors for metrics
        y_hat_m, y_m = self._prepare_image_metrics_inputs(y_hat, y)

        # Distortion metrics
        val_psnr = self.psnr(y_hat_m, y_m)
        val_ssim = self.ssim(y_hat_m, y_m)

        # Perceptual metrics (require 3 channels)
        y_hat_rgb, y_rgb = self._ensure_three_channels(y_hat_m, y_m)
        val_lpips = self.lpips(y_hat_rgb, y_rgb)
        val_dists = self.dists(y_hat_rgb, y_rgb)

        self.log(
            "val/psnr",
            val_psnr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_psnr", # To log in filename
            val_psnr,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/ssim",
            val_ssim,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/lpips",
            val_lpips,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/dists",
            val_dists,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
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

        y_hat_m, y_m = self._prepare_image_metrics_inputs(y_hat, y)

        psnr = self.psnr(y_hat_m, y_m)
        ssim = self.ssim(y_hat_m, y_m)

        y_hat_rgb, y_rgb = self._ensure_three_channels(y_hat_m, y_m)
        lpips = self.lpips(y_hat_rgb, y_rgb)
        dists = self.dists(y_hat_rgb, y_rgb)

        metrics = {
            "test_loss": test_loss,
            "test/psnr": psnr,
            "test/ssim": ssim,
            "test/lpips": lpips,
            "test/dists": dists,
        }
        
        if getattr(self, "diagnose_difficulty", False):
            diag_metrics = self._compute_difficulty_diagnostics(x, y, y_hat)
            metrics.update(diag_metrics)
        
        # center = x.shape[1] // 2

        # bicubic = F.interpolate(
        #     x[:, center],
        #     scale_factor=4,
        #     mode="bicubic",
        #     align_corners=False,
        # ).unsqueeze(1)

        # bicubic_psnr = self.psnr(
        #     bicubic.flatten(0, 1).clamp(0, 1),
        #     y.flatten(0, 1).clamp(0, 1),
        # )

        # gain = psnr - bicubic_psnr

        # print(
        #     batch_idx,
        #     "model:", psnr.item(),
        #     "bicubic:", bicubic_psnr.item(),
        #     "gain:", gain.item()
        # )
        
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
        self.log_dict(metrics, batch_size=batch_size, sync_dist=True)

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

        if y_hat.ndim == 5:
            b, t, c, h, w = y_hat.shape
            y_hat = y_hat.reshape(b * t, c, h, w)
            y = y.reshape(b * t, c, h, w)
            return y_hat, y

        if y_hat.ndim == 4:
            return y_hat, y

        raise ValueError(f"Unsupported tensor ndim for metrics: {y_hat.ndim}")

    def _prepare_image_metrics_inputs(self, y_hat, y):
        y_hat, y = self._flatten_video_for_metrics(y_hat, y)
        y_hat = y_hat.clamp(0, 1)
        y = y.clamp(0, 1)
        return y_hat, y

    def _ensure_three_channels(self, y_hat, y):
        if y_hat.ndim != 4 or y.ndim != 4:
            raise ValueError("Expected tensors with shape (N, C, H, W)")

        if y_hat.shape[1] == 1:
            y_hat = y_hat.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        if y_hat.shape[1] != 3:
            raise ValueError(
                f"LPIPS/DISTS require 1 or 3 channels, got {y_hat.shape[1]}"
            )

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
    
    
    
    def _psnr_manual(self, pred, target, eps=1e-10):
        """
        pred/target: (B,C,H,W) or (B,1,C,H,W)
        returns: scalar tensor averaged over batch
        """
        if pred.ndim == 5:
            pred = pred.flatten(0, 1)
            target = target.flatten(0, 1)

        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse + eps)
        return psnr.mean()


    def _laplacian_energy(self, x):
        """
        x: (B,C,H,W) or (B,1,C,H,W)
        """
        if x.ndim == 5:
            x = x.flatten(0, 1)

        k = x.new_tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ]).view(1, 1, 3, 3)

        k = k.repeat(x.shape[1], 1, 1, 1)
        y = F.conv2d(x, k, padding=1, groups=x.shape[1])
        return y.abs().mean()


    def _flow_magnitude_stats(self, flow):
        """
        flow: (B,T,2,H,W) or (B,2,H,W)
        """
        if flow.ndim == 5:
            mag = torch.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        elif flow.ndim == 4:
            mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        else:
            raise ValueError(f"Unsupported flow shape: {flow.shape}")

        return {
            "flow_mag_mean": mag.mean(),
            "flow_mag_max": mag.amax(),
            "flow_mag_std": mag.std(),
        }


    def _safe_flow_warp(self, img, flow):
        """
        img:  (B,C,H,W)
        flow: (B,2,H,W)
        """
        if not hasattr(self.model, "_warp"):
            return None
        return self.model._warp(img, flow)


    def _compute_difficulty_diagnostics(self, x, y, y_hat):
        """
        x:     (B,T,C,H,W), LQ
        y:     (B,1,C,sH,sW), GT center
        y_hat: (B,1,C,sH,sW), prediction center

        Returns dict of scalar tensors.
        """
        if x.ndim != 5:
            return {}

        B, T, C, H, W = x.shape
        center = T // 2

        x_center = x[:, center]

        # Bicubic baseline
        bicubic = F.interpolate(
            x_center,
            size=y.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).unsqueeze(1)

        model_psnr = self._psnr_manual(y_hat, y)
        bicubic_psnr = self._psnr_manual(bicubic, y)
        gain = model_psnr - bicubic_psnr

        # Temporal differences in LQ space
        diagnostics = {
            "diag/model_psnr": model_psnr.detach(),
            "diag/bicubic_psnr": bicubic_psnr.detach(),
            "diag/gain": gain.detach(),
            "diag/lap_lq_center": self._laplacian_energy(x_center).detach(),
            "diag/lap_gt": self._laplacian_energy(y).detach(),
            "diag/lap_pred": self._laplacian_energy(y_hat).detach(),
        }

        if center > 0:
            diagnostics["diag/temp_diff_prev"] = (
                x_center - x[:, center - 1]
            ).abs().mean().detach()

        if center + 1 < T:
            diagnostics["diag/temp_diff_next"] = (
                x_center - x[:, center + 1]
            ).abs().mean().detach()

        # Flow-based diagnostics, only if model exposes compute_flows_adjacent
        if hasattr(self.model, "compute_flows_adjacent") and T > 1:
            with torch.no_grad():
                flow_fwd, flow_bwd = self.model.compute_flows_adjacent(x)

            # flow_fwd[:, t] = t -> t+1
            # flow_bwd[:, t] = t+1 -> t
            if center > 0:
                flow_center_to_prev = flow_bwd[:, center - 1]
                stats_prev = self._flow_magnitude_stats(flow_center_to_prev)
                diagnostics.update({
                    f"diag/prev_{k}": v.detach()
                    for k, v in stats_prev.items()
                })

            if center + 1 < T:
                flow_center_to_next = flow_fwd[:, center]
                stats_next = self._flow_magnitude_stats(flow_center_to_next)
                diagnostics.update({
                    f"diag/next_{k}": v.detach()
                    for k, v in stats_next.items()
                })

            # Optional warping error in LQ space
            # prev -> center uses flow_fwd[:, center - 1]
            if center > 0:
                prev_to_center = flow_fwd[:, center - 1]
                warped_prev = self._safe_flow_warp(x[:, center - 1], prev_to_center)
                if warped_prev is not None:
                    diagnostics["diag/warp_error_prev"] = (
                        warped_prev - x_center
                    ).abs().mean().detach()

            # next -> center uses flow_bwd[:, center]
            if center + 1 < T:
                next_to_center = flow_bwd[:, center]
                warped_next = self._safe_flow_warp(x[:, center + 1], next_to_center)
                if warped_next is not None:
                    diagnostics["diag/warp_error_next"] = (
                        warped_next - x_center
                    ).abs().mean().detach()

        return diagnostics