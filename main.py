import argparse
import random
import os

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Argument Parser ===
def arg_creator():
    parser = argparse.ArgumentParser(description="COVID CT Classifier (PCA .pt version)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--pt_path", type=str, default=os.path.join("data", "PCA_Images"))
    parser.add_argument("--store_output", type=str, default=None)
    parser.add_argument("--conf_matrix", action="store_true")
    return parser.parse_args()

# === Dataset for PCA .pt files ===
class PCAFileDataset(Dataset):
    def __init__(self, pca_root_dir):
        self.samples = []
        self.pca_root_dir = pca_root_dir

        for label_name, label in [("non-COVID", 0), ("COVID", 1)]:
            label_dir = os.path.join(pca_root_dir, label_name)
            for fname in os.listdir(label_dir):
                if fname.endswith(".pt"):
                    full_path = os.path.join(label_name, fname)
                    self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(self.pca_root_dir, rel_path)
        tensor = torch.load(full_path)
        return tensor.float(), label

# === Model ===
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# === Training and Evaluation ===
def train_model(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct = 0.0, 0
    for x, label in tqdm(loader, desc=f"Train Epoch {epoch}"):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == label).sum().item()
    acc = 100. * correct / len(loader.dataset)
    return acc, total_loss / len(loader)

def eval_model(model, loader, criterion, epoch):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, label in tqdm(loader, desc=f"Eval Epoch {epoch}"):
            x, label = x.to(device), label.to(device)
            output = model(x)
            loss = criterion(output, label)
            total_loss += loss.item()
            correct += (output.argmax(1) == label).sum().item()
    acc = 100. * correct / len(loader.dataset)
    return acc, total_loss / len(loader)

# === Confusion Matrix ===
def confusion_matrix_gen(model, dataset, class_names):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            preds = model(x).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    cm = pd.DataFrame(
        confusion_matrix(all_labels, all_preds),
        index=class_names, columns=class_names
    )
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf")

# === Training Loop ===
def train_val_loop(model):
    dataset = PCAFileDataset(args.pt_path)
    train_size = int(args.train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")
    if args.store_output:
        stats = []

    for epoch in range(1, args.epochs + 1):
        train_acc, train_loss = train_model(model, train_loader, optimizer, criterion, epoch)
        val_acc, val_loss = eval_model(model, val_loader, criterion, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/simpleclassifier_epoch{epoch}.pth")

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if args.store_output:
            stats.append({"epoch": epoch, "train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})

    if args.store_output:
        pd.DataFrame(stats).to_csv(args.store_output + ".csv", index=False)

# === Main ===
def main():
    global args, device
    args = arg_creator()
    device = torch.device(args.device)

    try:
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device not available.")
    except:
        print("MPS not available, falling back to CPU.")
        device = torch.device("cpu")

    # Determine input dim from one .pt file
    sample_path = os.path.join(args.pt_path, "COVID")
    for fname in os.listdir(sample_path):
        if fname.endswith(".pt"):
            example_tensor = torch.load(os.path.join(sample_path, fname))
            break
    input_dim = example_tensor.shape[0]

    model = SimpleClassifier(input_dim=input_dim, num_classes=2).to(device)

    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))

    if args.conf_matrix:
        dataset = PCAFileDataset(args.pt_path)
        confusion_matrix_gen(model, dataset, ["Non-COVID", "COVID"])
    else:
        train_val_loop(model)

if __name__ == "__main__":
    main()