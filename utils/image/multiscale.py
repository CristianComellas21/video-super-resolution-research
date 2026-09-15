# Description: Functions to work with multiple scale images based in the 
# implementation of the paper "Multi-Scale DCT Denoising" doi:http://dx.doi.org/10.1137/17M1111826

from utils.image import torch_dct
import torch

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def extract_scale(img, scale):
    """
    Extract a scale from an image.
    Args:
        img: Image to extract scale from. Shape [B, C, H, W].
        scale: Scale to extract.
    Returns:
        Image at the specified scale.
    """
    if scale == 1:
        return img
    
    
    h, w = img.shape[-2:]
    h_out = h // (2 ** (scale - 1))
    w_out = w // (2 ** (scale - 1))
    
    freq = torch_dct.dct_2d(img)
    
    scaling = torch.sqrt(torch.tensor((w_out * h_out) / (w * h)).to(img.device).to(img.dtype))
    
    result = freq[..., :h_out, :w_out] * scaling
    
    return torch_dct.idct_2d(result)



def merge_coarse(image, coarse, f_rec):
    
    if not isinstance(f_rec, torch.Tensor):
        f_rec = torch.tensor(f_rec)
    
    freq = torch_dct.dct_2d(image)
    tmp = torch_dct.dct_2d(coarse)
    
    h, w = coarse.shape[-2:]
    w_out, h_out = image.shape[-2:]
    
    h_rec = FloorSTE.apply((h * f_rec[0]).to(image.device)).to(torch.int64)
    w_rec = FloorSTE.apply((w * f_rec[1]).to(image.device)).to(torch.int64)
    
    scaling = torch.sqrt(torch.tensor((w_out * h_out) / (w * h)).to(image.device).to(image.dtype))
    
    freq[..., :h_rec, :w_rec] = tmp[..., :h_rec, :w_rec] * scaling
    
    return torch_dct.idct_2d(freq)