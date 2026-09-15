import torch
import torch.nn.functional as F
import numpy as np

from skimage import io
from pathlib import Path
from argparse import ArgumentParser


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def img_to_batched_tensor(img: np.ndarray) -> torch.Tensor:
    # Convert image to tensor (H, W, C) -> (1, C, H, W) in [0, 1]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    return (tensor / 255.0).to(DEVICE)


def batched_tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
    # Convert tensor (1, C, H, W) in [0, 1] -> image (H, W, C) uint8
    tensor = tensor.clamp(0.0, 1.0)
    out = (tensor.permute(0, 2, 3, 1).squeeze(0) * 255.0).byte().cpu().numpy()

    # If grayscale, return (H, W) for skimage compatibility
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    return out


def upsample_bicubic(img: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(
        img,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    if args.scale < 1:
        raise ValueError(f"scale must be >= 1, got {args.scale}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = io.imread(args.img)
    img_name = Path(args.img).stem

    img_tensor = img_to_batched_tensor(img)

    with torch.no_grad():
        upsampled = upsample_bicubic(img_tensor, args.scale)

    upsampled_img = batched_tensor_to_img(upsampled)

    io.imsave(output_dir / f"bicubic_x{args.scale}_{img_name}.png", upsampled_img)
    # io.imsave(output_dir / f"input_{img_name}.png", img)


if __name__ == "__main__":
    main()
