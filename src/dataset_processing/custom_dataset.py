import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Ensure image is a NumPy array before applying Albumentations (if it's not a tensor)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Convert torch tensor to NumPy array (C, H, W -> H, W, C)
        elif isinstance(image, Image.Image):
            image = np.array(image)  # Convert PIL Image to NumPy array

        # Apply the transformation if provided
        if self.transform:
            image = self.transform(image=image)["image"]

        # Convert back to tensor if needed (after transformation)
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(2, 0, 1)  # Convert to tensor (H, W, C -> C, H, W)

        return image, label
