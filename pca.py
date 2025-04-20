import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle

# === Argument Parser ===
def arg_parser():
    parser = argparse.ArgumentParser(description="Apply PCA to CT Images and save results")
    parser.add_argument("--input_dir", type=str, default=os.path.join("data", "images"), help="Path to original image directory")
    parser.add_argument("--output_dir", type=str, default="data/PCA_images", help="Path to save PCA transformed tensors")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--max_width", type=int, default=224)
    parser.add_argument("--max_height", type=int, default=224)
    return parser.parse_args()

# === Main Processing ===
def main():
    args = arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.max_width, args.max_height), antialias=True),
        transforms.ToTensor()
    ])

    data = []
    filenames = []

    for label_folder in ["non-COVID", "COVID"]:
        full_path = os.path.join(args.input_dir, label_folder)
        for fname in tqdm(os.listdir(full_path), desc=f"Processing {label_folder}"):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(full_path, fname)
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                data.append(image.view(-1).numpy())
                filenames.append((label_folder, fname))

    data_matrix = np.stack(data)
    print(f"Shape before PCA: {data_matrix.shape}")

    pca = PCA(n_components=args.n_components)
    pca_data = pca.fit_transform(data_matrix)
    print(f"Shape after PCA: {pca_data.shape}")

    # Save transformed data
    for i, (label_folder, fname) in enumerate(filenames):
        save_path = os.path.join(args.output_dir, label_folder)
        os.makedirs(save_path, exist_ok=True)
        file_out = os.path.join(save_path, fname.replace(".jpg", ".pt").replace(".jpeg", ".pt").replace(".png", ".pt"))
        torch.save(torch.tensor(pca_data[i]), file_out)

    # Save PCA model for later inverse_transform or transform
    with open(os.path.join(args.output_dir, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print("PCA processing complete.")

if __name__ == "__main__":
    main()
