import torch

# Matrix for color transform
YUV_MATRIX = [
    [0.5773502588272094726562500000000000000000,
      0.5773502588272094726562500000000000000000,
      0.5773502588272094726562500000000000000000],
     #
     [0.7071067690849304199218750000000000000000,
      0.0000000000000000000000000000000000000000,
      -0.7071067690849304199218750000000000000000],
     #
     [0.4082483053207397460937500000000000000000,
      -0.8164966106414794921875000000000000000000,
      0.4082483053207397460937500000000000000000]
     ]

def YUV_forward(img_rgb):
    
    r = torch.squeeze(img_rgb[:,0,:,:])
    g = torch.squeeze(img_rgb[:,1,:,:])
    b = torch.squeeze(img_rgb[:,2,:,:])
    
    out = torch.zeros(img_rgb.size(), device=img_rgb.device)        
    out[:,0,:,:] = YUV_MATRIX[0][0]*r + YUV_MATRIX[0][1]*g + YUV_MATRIX[0][2]*b
    out[:,1,:,:] = YUV_MATRIX[1][0]*r + YUV_MATRIX[1][1]*g + YUV_MATRIX[1][2]*b
    out[:,2,:,:] = YUV_MATRIX[2][0]*r + YUV_MATRIX[2][1]*g + YUV_MATRIX[2][2]*b
    
    return out


def YUV_reverse(img):
        
    img1 = torch.squeeze(img[:,0,:,:])
    img2 = torch.squeeze(img[:,1,:,:])
    img3 = torch.squeeze(img[:,2,:,:])
    
    img_rgb = torch.zeros(img.size(), device=img.device)       
    img_rgb[:,0,:,:] = YUV_MATRIX[0][0]*img1 + YUV_MATRIX[1][0]*img2 + YUV_MATRIX[2][0]*img3
    img_rgb[:,1,:,:] = YUV_MATRIX[0][1]*img1 + YUV_MATRIX[1][1]*img2 + YUV_MATRIX[2][1]*img3
    img_rgb[:,2,:,:] = YUV_MATRIX[0][2]*img1 + YUV_MATRIX[1][2]*img2 + YUV_MATRIX[2][2]*img3
    
    return img_rgb