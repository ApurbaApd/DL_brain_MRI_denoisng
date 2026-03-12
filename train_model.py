import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess import MedicalDenoisingDataset, MedicalDenoisingValidationDataset

# from models import CAE, UNet, VAE, ResNetAE
from models import UNet, ResNetAE, CAE

def extract_patient_id(filename):
    return filename.split("_")[0]


def get_model(name, device):
    if name == 'cae': return CAE().to(device)
    if name == 'unet': return UNet().to(device)
    # if name == 'vae': return VAE().to(device)
    if name == 'resnet': return ResNetAE().to(device)
    raise ValueError(f"Unknown model: {name}")


def train(args):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"--- Training {args.model.upper()} on {device} ---")

    files = [f for f in os.listdir(args.data_path) if f.endswith(".png")]

    data = pd.DataFrame({"filename": files})
    data["patient_id"] = data["filename"].apply(extract_patient_id)

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    PATIENCE = 5

    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(data, groups=data["patient_id"])):

        print(f"\n====Fold {fold+1}: =====")

        train_files = data.iloc[tr_idx]["filename"].values
        val_files = data.iloc[val_idx]["filename"].values

        train_dataset = MedicalDenoisingDataset(train_files, args.data_path)
        val_dataset = MedicalDenoisingValidationDataset(val_files, args.data_path)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = get_model(args.model, device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        early_stop_counter = 0
        val_loss_history = []
        train_loss_history = []

        for epoch in range(NUM_EPOCHS):

            # Train
            model.train()
            train_loss = 0

            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)

                optimizer.zero_grad()
                output = model(noisy)
                loss = criterion(output, clean)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_loss_history.append(train_loss)

            # validation
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    output = model(noisy)
                    loss = criterion(output, clean)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}: "
                  f"Train={train_loss:.4f} | Val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(),
                           f"best_model_vary_{args.model}_fold{fold+1}.pth")
                        #    f"best_model_{args.model}_fold{fold+1}.pth")
            else:
                early_stop_counter += 1

            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
        print(f"Mean Train Loss: {np.mean(train_loss_history):.4f} | Mean Val Loss: {np.mean(val_loss_history):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['cae', 'unet', 'vae', 'resnet'])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train(args)
