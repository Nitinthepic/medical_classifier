import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

# === Config ===
CHECKPOINT_PATH = "checkpoints/simpleclassifier_epoch10.pth"
PCA_MODEL_PATH = os.path.join("data", "PCA_Images", "pca_model.pkl")
TOP_K = 5
IMAGE_SHAPE = (3, 224, 224)  

with open(PCA_MODEL_PATH, "rb") as f:
    pca = pickle.load(f)  

class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

input_dim = pca.components_.shape[0]  # num PCA components
model = SimpleClassifier(input_dim=input_dim, num_classes=2)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()


with torch.no_grad():
    weights = model.model[0].weight  # shape [128, input_dim]
    importance = weights.abs().mean(dim=0)  # shape [input_dim]
    topk_indices = torch.topk(importance, k=TOP_K).indices

components = pca.components_  # shape [n_components, original_dim]

os.makedirs("eigenimages", exist_ok=True)

for i, idx in enumerate(topk_indices):
    eigenvector = components[idx]  
    reshaped = eigenvector.reshape(IMAGE_SHAPE)
    grayscale = reshaped.mean(axis=0) 

    plt.figure(figsize=(4, 4))
    plt.imshow(grayscale, cmap="viridis")
    plt.colorbar()
    plt.title(f"Principal Component {idx.item()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"eigenimages/eigen_pc{idx.item():03d}.png")
    plt.close()

    print(f"Saved heatmap for PC{idx.item()} as eigen_pc{idx.item():03d}.png")

print("Done. Saved top eigenimage heatmaps to ./eigenimages/")