import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

def get_epi(clean, denoised):
    """Edge Preservation Index: Correlations of Laplacian Edges"""
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(clean.device)
    clean_edges = F.conv2d(clean, kernel, padding=1)
    denoised_edges = F.conv2d(denoised, kernel, padding=1)
    
    # Flatten and Cosine Similarity
    return F.cosine_similarity(clean_edges.flatten(1), denoised_edges.flatten(1)).mean().item()

def get_metrics_batch(clean, denoised):
    """Calculates metrics for a whole batch"""
    c_np = clean.cpu().detach().numpy()
    d_np = denoised.cpu().detach().numpy()
    
    batch_psnr = []
    batch_ssim = []
    
    for i in range(c_np.shape[0]):
        p = psnr_metric(c_np[i,0], d_np[i,0], data_range=1.0)
        s = ssim_metric(c_np[i,0], d_np[i,0], data_range=1.0)
        batch_psnr.append(p)
        batch_ssim.append(s)
        
    return {
        'PSNR': sum(batch_psnr)/len(batch_psnr), 
        'SSIM': sum(batch_ssim)/len(batch_ssim), 
        'EPI': get_epi(clean, denoised)
    }