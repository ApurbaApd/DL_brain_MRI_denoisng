import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MedicalDenoisingDataset(Dataset):
    def __init__(self, file_list, root_dir, img_size=128):

        self.root_dir = root_dir
        self.img_size = img_size
        self.files = [os.path.join(root_dir, f) for f in file_list]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = self.files[idx]

        try:
            clean = Image.open(img_path).convert('L')
            clean = self.transform(clean)

            # Random sigma between 0.05 and 0.1
            sigma = torch.empty(1).uniform_(0.01, 0.1).item()

            # Gaussian noise
            noise = torch.randn_like(clean) * sigma
            noisy = torch.clamp(clean + noise, 0., 1.)

            return noisy, clean

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(1, self.img_size, self.img_size), \
                   torch.zeros(1, self.img_size, self.img_size)


class MedicalDenoisingValidationDataset(Dataset):
    def __init__(self, file_list, root_dir, noise_factor=0.01, img_size=128):

        self.root_dir = root_dir
        self.noise_factor = noise_factor
        self.img_size = img_size

        # full paths from file list
        self.files = [os.path.join(root_dir, f) for f in file_list]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = self.files[idx]

        try:
            clean = Image.open(img_path).convert('L')
            clean = self.transform(clean)

            # Add Gaussian Noise
            noise = torch.randn_like(clean) * self.noise_factor
            # sigma = torch.empty(1).uniform_(0.05, 0.2).item()
            # noise = torch.randn_like(clean) * sigma
            noisy = torch.clamp(clean + noise, 0., 1.)

            return noisy, clean

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(1, self.img_size, self.img_size), \
                   torch.zeros(1, self.img_size, self.img_size)