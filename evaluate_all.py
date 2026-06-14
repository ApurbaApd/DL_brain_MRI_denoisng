import os
import torch
from torch.utils.data import DataLoader
from models import UNet, ResNetAE, CAE
from preprocess import MedicalDenoisingDataset
from metrics import get_metrics_batch


def evaluate_model():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Evaluating on {device}")

    test_path = "test_images-2"

    test_files = [f for f in os.listdir(test_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    test_dataset = MedicalDenoisingDataset(test_files, test_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # model = ResNetAE().to(device)
    model = CAE().to(device)
    model.load_state_dict(torch.load("best_model_cae_fold5.pth",
                                     map_location=device))
    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_epi = 0
    total_images = 0

    with torch.no_grad():
        for noisy, clean in test_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            metrics = get_metrics_batch(clean, output)

            batch_size = noisy.size(0)

            total_psnr += metrics['PSNR'] * batch_size
            total_ssim += metrics['SSIM'] * batch_size
            total_epi += metrics['EPI'] * batch_size
            total_images += batch_size

    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    avg_epi = total_epi / total_images

    print("\n FINAL TEST RESULTS")
    print(f"PSNR : {avg_psnr:.2f} dB")
    print(f"SSIM : {avg_ssim:.4f}")
    print(f"EPI  : {avg_epi:.4f}")


if __name__ == "__main__":
    evaluate_model()