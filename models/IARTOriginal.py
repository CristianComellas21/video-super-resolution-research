from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Mapping
import sys

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


class OutputMode(str, Enum):
    CENTER = "center"
    ALL = "all"


class PretrainedVariant(str, Enum):
    REDS_N6 = "reds_n6"
    REDS_N16 = "reds_n16"
    VIMEO_BI = "vimeo_bi"
    VIMEO_BD = "vimeo_bd"


@ARCH_REGISTRY.register()
class Model(nn.Module):
    """Wrapper del IART original para integrarlo con la firma del proyecto."""

    _VARIANT_TO_FILE: Mapping[str, str] = {
        PretrainedVariant.REDS_N6.value: "IART_REDS_BI_N6.pth",
        PretrainedVariant.REDS_N16.value: "IART_REDS_BI_N16.pth",
        PretrainedVariant.VIMEO_BI.value: "IART_Vimeo-90K_BI.pth",
        PretrainedVariant.VIMEO_BD.value: "IART_Vimeo-90K_BD.pth",
    }

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        embed_dim: int = 120,
        depths: tuple[int, ...] = (6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6),
        window_size: tuple[int, ...] = (3, 8, 8),
        num_frames: int = 3,
        img_size: int = 64,
        patch_size: int = 1,
        cpu_cache_length: int = 100,
        is_low_res_input: bool = True,
        use_checkpoint: tuple[bool, ...] = (True, True, True, True),
        output_mode: str = OutputMode.CENTER.value,
        iart_repo_dir: str | None = None,
        spynet_path: str | None = None,
        load_pretrained: bool = True,
        pretrained_variant: str = PretrainedVariant.REDS_N16.value,
        pretrained_path: str | None = None,
        strict_load: bool = True,
    ):
        super().__init__()

        self.output_mode = OutputMode(output_mode)

        workspace_root = Path(__file__).resolve().parents[2]
        iart_root = Path(iart_repo_dir) if iart_repo_dir is not None else workspace_root / "IART"
        if not iart_root.exists():
            raise FileNotFoundError(f"No existe iart_repo_dir: {iart_root}")

        if str(iart_root) not in sys.path:
            sys.path.insert(0, str(iart_root))

        from archs.iart_arch import IART as IARTOriginalArch

        spynet_ckpt = (
            Path(spynet_path)
            if spynet_path is not None
            else iart_root / "experiments" / "pretrained_models" / "flownet" / "spynet_sintel_final-3d2a1287.pth"
        )

        self.model = IARTOriginalArch(
            in_channels=in_channels,
            mid_channels=mid_channels,
            embed_dim=embed_dim,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            window_size=tuple(window_size),
            num_frames=num_frames,
            img_size=img_size,
            patch_size=patch_size,
            cpu_cache_length=cpu_cache_length,
            is_low_res_input=is_low_res_input,
            use_checkpoint=list(use_checkpoint),
            spynet_path=str(spynet_ckpt),
        )

        if not bool(load_pretrained):
            return

        if pretrained_path is None:
            variant = PretrainedVariant(pretrained_variant)
            pretrained_ckpt = iart_root / "experiments" / "pretrained_models" / self._VARIANT_TO_FILE[variant.value]
        else:
            pretrained_ckpt = Path(pretrained_path)

        self._load_pretrained(pretrained_ckpt, strict=bool(strict_load))

    def _extract_state_dict(self, raw):
        if not isinstance(raw, dict):
            return raw

        candidate_keys = [
            "params_ema",
            "params",
            "state_dict",
            "model_state_dict",
            "net_g",
            "network_g",
            "model",
        ]
        for key in candidate_keys:
            value = raw.get(key)
            if isinstance(value, dict):
                return value
        return raw

    def _load_pretrained(self, ckpt_path: Path, strict: bool) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No existe checkpoint de IART: {ckpt_path}")

        raw = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = self._extract_state_dict(raw)

        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Formato de checkpoint no soportado en {ckpt_path}")

        cleaned = {}
        for k, v in state_dict.items():
            if not isinstance(k, str):
                cleaned[k] = v
                continue
            key = k
            if key.startswith("module."):
                key = key[7:]
            if key.startswith("net_g."):
                key = key[6:]
            cleaned[key] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=strict)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                "Error cargando pesos de IART en modo estricto. "
                f"Missing: {missing[:20]} | Unexpected: {unexpected[:20]}"
            )

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        out = self.model(lqs)

        if self.output_mode == OutputMode.ALL:
            return out

        # Alineado con el resto de modelos del repo: salida temporal centrada.
        t = out.shape[1] // 2
        return out[:, t:t + 1]