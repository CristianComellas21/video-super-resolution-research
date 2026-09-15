from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import flow_warp



import torch
import numpy as np

from skimage import io
from pathlib import Path
from argparse import ArgumentParser


SPYNET_PATH = "/home/cristiancomellas/phd/super-resolution/code/checkpoints/flow/spynet/spynet_sintel_final-3d2a1287.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # flow_warp expects (B, H, W, 2)
    return flow_warp(img, flow.permute(0, 2, 3, 1))

def img_to_batched_tensor(img: np.ndarray) -> torch.Tensor:
    # Convert image to tensor (B, H, W, C) -> (B, C, H, W)
    tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    return (tensor / 255.0).to(DEVICE)

def batched_tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
    # Convert tensor to image (B, C, H, W) -> (B, H, W, C)
    tensor = tensor.clamp(0.0, 1.0)
    return (tensor.permute(0, 2, 3, 1).squeeze() * 255.0).byte().cpu().numpy()

def main():
    
    # Load paremters
    parser = ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SpyNet
    flownet = SpyNet(SPYNET_PATH).to(DEVICE)
    flownet.eval()
    
    # Load images
    ref = io.imread(args.ref)
    img = io.imread(args.img)

    if ref.shape != img.shape:
        raise ValueError(f"ref and img must have the same shape, got {ref.shape} and {img.shape}")
    
    # Get ref and image name for saving outputs
    ref_name = args.ref.split("/")[-1].split(".")[0]
    img_name = args.img.split("/")[-1].split(".")[0]
    
    # Convert to tensors
    ref_tensor = img_to_batched_tensor(ref)  # (H, W, C) -> (1, C, H, W)
    img_tensor = img_to_batched_tensor(img)  # (H, W, C) -> (1, C, H, W)

    # Compute flow from ref to img
    with torch.no_grad():
        flow = flownet(ref_tensor, img_tensor)
        
    # Warp img to ref
    warped = warp(img_tensor, flow)
    warped_img = batched_tensor_to_img(warped)
    

    # Save warped image
    io.imsave(output_dir / f"warped_{img_name}_on_{ref_name}.png", warped_img)
    
    # Save flow as .tiff
    flow_np = flow.squeeze().permute(1, 2, 0).cpu().numpy()  # (1, 2, H, W) -> (H, W, 2)
    flow_u = flow_np[..., 0]
    flow_v = flow_np[..., 1]
    flow_rgb = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.float32)
    flow_rgb[..., 0] = (flow_u - flow_u.min()) / (flow_u.max() - flow_u.min() + 1e-8) * 255  # Normalize to [0, 255]
    flow_rgb[..., 1] = (flow_v - flow_v.min()) / (flow_v.max() - flow_v.min() + 1e-8) * 255  # Normalize to [0, 255]
    io.imsave(output_dir / f"flow_{ref_name}_to_{img_name}.tiff", flow_rgb.astype(np.uint8))
    
    
    # Save original images for reference in output dir
    io.imsave(output_dir / f"ref_{ref_name}.png", ref)
    io.imsave(output_dir / f"img_{img_name}.png", img)


if __name__ == "__main__":
    main()