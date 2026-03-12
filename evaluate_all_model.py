import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import CAE, VAE, UNet, ResNetAE
from metrics import get_metrics_batch
from preprocess import MedicalDenoisingDataset, MedicalDenoisingValidationDataset
import os

def evaluate_all():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # 2. Check for Apple GPU (MacOS)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    # 3. Fallback to CPU
    else:
        device = torch.device('cpu')
    
    # 1. Load Test Data
    test_path = "test_images-2"

    if not os.path.exists(test_path):
        print("Test folder not found!")
        return

    test_files = [f for f in os.listdir(test_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    test_dataset = MedicalDenoisingValidationDataset(test_files, test_path)
    loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)
    
    # 2. Load Models
    models = {
        # 'CAE': CAE(),
        'UNet': UNet(),
        # 'VAE': VAE(),
        'ResNet': ResNetAE()
    }
    
    results = {}
    print(f"{'Model':<10} | {'PSNR':<10} | {'SSIM':<10} | {'EPI':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        path = f"best_model_vary_{name.lower()}_fold3.pth"
        if not os.path.exists(path):
            print(f"Skipping {name} (Weights not found)")
            continue
            
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        
        with torch.no_grad():
            if name == 'VAE':
                out, _, _ = model(noisy)
            else:
                out = model(noisy)
            
            metrics = get_metrics_batch(clean, out)
            print(f"{name:<10} | {metrics['PSNR']:.2f} dB   | {metrics['SSIM']:.4f}     | {metrics['EPI']:.4f}")
            #save output for visualization
            # os.makedirs("results", exist_ok=True)
            # torch.save(out, f"results/{name}_metrics.pdf")
            
            results[name] = out
            
    # 3. Visualization
    fig, axes = plt.subplots(1, len(results) + 2, figsize=(20, 8), dpi=300)
    
    # Plot Original & Noisy
    axes[0].imshow(clean[0].squeeze().cpu(), cmap='gray'); axes[0].set_title("Ground Truth"); axes[0].axis('off')
    axes[1].imshow(noisy[0].squeeze().cpu(), cmap='gray'); axes[1].set_title("Noisy Input"); axes[1].axis('off')
    
    # Plot Models
    for i, (name, out_tensor) in enumerate(results.items()):
        ax = axes[i+2]
        ax.imshow(out_tensor[0].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(f"{name}")
        ax.axis('off')
        
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/final_benchmark.png")

    print("\nBenchmark saved to results/final_benchmark.png")
    

if __name__ == "__main__":
    evaluate_all()