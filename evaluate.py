import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import UNet, ResNetAE, CAE, VAE
# from models import CAE, UNet, VAE, ResNetAE
from preprocess import MedicalDenoisingDataset, MedicalDenoisingValidationDataset
from metrics import get_metrics_batch


def evaluate_all():

 
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Evaluating on {device}")

    test_path = "test_images-2"

    if not os.path.exists(test_path):
        print("Test folder not found!")
        return

    test_files = [f for f in os.listdir(test_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    test_dataset = MedicalDenoisingValidationDataset(test_files, test_path)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)


    # models = {
    #     'CAE': CAE(),
    #     'UNet': UNet(),
    #     'VAE': VAE(),
    #     'ResNet': ResNetAE()
    # }
    
    models = {
        'UNet': UNet(),
        # 'ResNet': ResNetAE(),
        # 'CAE': CAE(),
        # 'VAE': VAE(),
    }


    results = {}

    print(f"\n{'Model':<10} | {'PSNR':<10} | {'SSIM':<10} | {'EPI':<10}")
    print("-" * 55)

    for name, model in models.items():

        # Adjust this if your saved path is different
        weight_path = f"best_model_{name.lower()}_fold1.pth"

        if not os.path.exists(weight_path):
            print(f"Skipping {name} (weights not found)")
            continue

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():

            if name == 'VAE':
                output, _, _ = model(noisy)
            else:
                output = model(noisy)

            metrics = get_metrics_batch(clean, output)

            print(f"{name:<10} | "
                  f"{metrics['PSNR']:.2f} dB   | "
                  f"{metrics['SSIM']:.4f}     | "
                  f"{metrics['EPI']:.4f}")

            results[name] = output

 
    fig, axes = plt.subplots(1, len(results) + 2, figsize=(18, 7), dpi=300)

    axes[0].imshow(clean[0].squeeze().cpu(), cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(noisy[0].squeeze().cpu(), cmap='gray')
    axes[1].set_title("Noisy Input")
    axes[1].axis('off')

    for i, (name, out_tensor) in enumerate(results.items()):
        ax = axes[i + 2]
        ax.imshow(out_tensor[0].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/final_benchmark.png")

    print("\nBenchmark saved to results/final_benchmark.png")


if __name__ == "__main__":
    evaluate_all()