import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class PersonAttributeDataset(Dataset):
    def __init__(self, txt_file, img_dir):
        self.img_dir = img_dir
        with open(txt_file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx].strip().split()

        # Add extension if needed
        img_name = row[0] + ".jpg"

        labels = np.array(row[1:], dtype=np.float32)

        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.resize(image, (224, 224))
        image = image / 255.0

        image = torch.tensor(image).permute(2, 0, 1).float()
        labels = torch.tensor(labels)

        return image, labels