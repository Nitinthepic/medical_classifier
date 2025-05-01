import os
import torch
import pickle
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# === Config ===
PCA_MODEL_PATH = os.path.join("data", "PCA_Images", "pca_model.pkl")
PCA_FILES_DIR = os.path.join("data", "PCA_Images")
OUTPUT_DIR = os.path.join("data", "Reconstructed_Images")
IMAGE_SHAPE = (3, 224, 224)  # original shape before PCA (channels, height, width)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load PCA model ===
with open(PCA_MODEL_PATH, "rb") as f:
    pca = pickle.load(f)

# === Reconstruct and save ===
for label_folder in ["non-COVID", "COVID"]:
    input_folder = os.path.join(PCA_FILES_DIR, label_folder)
    output_folder = os.path.join(OUTPUT_DIR, label_folder)
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.endswith(".pt"):
            pca_tensor = torch.load(os.path.join(input_folder, fname))
            vector = pca_tensor.numpy().reshape(1, -1)
            reconstructed_flat = pca.inverse_transform(vector).reshape(-1)

            image_tensor = torch.tensor(reconstructed_flat).view(*IMAGE_SHAPE).clamp(0, 1)
            img = to_pil_image(image_tensor)
            img.save(os.path.join(output_folder, fname.replace(".pt", ".png")))

print("Reconstruction complete. Saved images to:", OUTPUT_DIR)