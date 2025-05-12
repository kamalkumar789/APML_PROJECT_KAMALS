import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform_label0=None, transform_label1=None):
        self.base_dataset = base_dataset
        self.transform_label0 = transform_label0
        self.transform_label1 = transform_label1

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # Convert PIL to NumPy if using albumentations
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] → [H, W, C]
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        if label == 0 and self.transform_label0 is not None:
            # Assume transform_label0 is torchvision transform
            image = self.transform_label0(image=image_np)["image"]
        elif label == 1 and self.transform_label1 is not None:
            # Assume transform_label1 is albumentations transform
            image = self.transform_label1(image=image_np)["image"]

        return image, label

    def __len__(self):
        return len(self.base_dataset)