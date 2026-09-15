from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        size: str = "large",
        custom_path: str | None = None,
        progress: bool = True,
        force_fp32: bool = True,
        min_height: int = 64,
        min_width: int = 128,
    ):
        super().__init__()

        try:
            from torchvision.models.optical_flow import (
                Raft_Large_Weights,
                Raft_Small_Weights,
                raft_large,
                raft_small,
            )
        except Exception as exc:
            raise RuntimeError(
                "No pude importar RAFT desde torchvision. "
                "Necesitas torchvision>=0.14 con modelos optical_flow."
            ) from exc

        self.size = str(size).lower()
        if self.size not in {"large", "small"}:
            raise ValueError(f"size debe ser 'large' o 'small', recibido: {size}")

        if self.size == "small":
            default_weights = Raft_Small_Weights.DEFAULT
            model_builder = raft_small
        else:
            default_weights = Raft_Large_Weights.DEFAULT
            model_builder = raft_large

        # Si hay checkpoint custom, evitamos descargar pesos y luego cargamos el estado.
        if custom_path is None:
            self.model = model_builder(weights=default_weights, progress=progress)
        else:
            self.model = model_builder(weights=None, weights_backbone=None, progress=progress)
            self._load_custom_checkpoint(Path(custom_path))

        # Reutilizamos el preprocess oficial según tamaño.
        self.preprocess = default_weights.transforms()
        self.force_fp32 = bool(force_fp32)
        self.min_height = int(min_height)
        self.min_width = int(min_width)
        if self.min_height < 0:
            raise ValueError(f"min_height debe ser >= 0, recibido: {min_height}")
        if self.min_width < 0:
            raise ValueError(f"min_width debe ser >= 0, recibido: {min_width}")

    def _load_custom_checkpoint(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No existe custom_path para RAFT: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
            for key in ("state_dict", "model", "params"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            state_dict = {
                (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(
                "Error cargando custom_path en RAFT. "
                f"Missing keys: {missing[:10]} | Unexpected keys: {unexpected[:10]}"
            )

    @staticmethod
    def _pad_to_multiple_of_8(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        _, _, h, w = a.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h == 0 and pad_w == 0:
            return a, b, 0, 0

        a_pad = F.pad(a, (0, pad_w, 0, pad_h), mode="replicate")
        b_pad = F.pad(b, (0, pad_w, 0, pad_h), mode="replicate")
        return a_pad, b_pad, pad_h, pad_w

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.shape != img2.shape:
            raise ValueError(f"img1 e img2 deben tener mismo shape, got {tuple(img1.shape)} vs {tuple(img2.shape)}")

        orig_h, orig_w = img1.shape[-2:]
        work_h, work_w = orig_h, orig_w
        # torchvision RAFT puede devolver NaN en resoluciones muy estrechas.
        # En particular, con W < 128 se reproduce consistentemente en este entorno.
        work_h = max(orig_h, self.min_height)
        work_w = max(orig_w, self.min_width)

        resized = (work_h != orig_h) or (work_w != orig_w)
        if resized:
            img1 = F.interpolate(img1, size=(work_h, work_w), mode="bilinear", align_corners=False)
            img2 = F.interpolate(img2, size=(work_h, work_w), mode="bilinear", align_corners=False)

        scale_x = work_w / float(orig_w)
        scale_y = work_h / float(orig_h)

        if self.force_fp32:
            # RAFT de torchvision puede ser inestable en bf16/fp16 dentro de autocast.
            device_type = "cuda" if img1.is_cuda else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                flow = self._forward_impl(img1.float(), img2.float())
        else:
            flow = self._forward_impl(img1, img2)

        if resized:
            flow = F.interpolate(flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            flow[:, 0, :, :] = flow[:, 0, :, :] / scale_x
            flow[:, 1, :, :] = flow[:, 1, :, :] / scale_y

        if not torch.isfinite(flow).all():
            raise RuntimeError("RAFT produjo NaN/Inf en el flujo. Revisa input range, precision y checkpoint.")
        return flow

    def _forward_impl(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        _, _, h, w = img1.shape
        img1_pad, img2_pad, _, _ = self._pad_to_multiple_of_8(img1, img2)
        img1_pp, img2_pp = self.preprocess(img1_pad, img2_pad)
        flow_list = self.model(img1_pp, img2_pp)
        flow = flow_list[-1]
        return flow[:, :, :h, :w]
