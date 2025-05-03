import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImagesDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform_for_label_0=None, transform_for_label_1=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)

        self.image_names = self.data['image_name'].tolist()
        self.labels = self.data['target'].tolist()

        self.transform_label_zero = transform_for_label_0
        self.transform_label_one = transform_for_label_1

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]

        image_path = os.path.join(self.image_folder, image_name)
        if not image_path.lower().endswith('.jpg'):
            image_path += ".jpg"

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Apply class-specific transforms
        if label == 0 and self.transform_label_zero:
            image = self.transform_label_zero(image)
        elif label == 1 and self.transform_label_one:
            image = self.transform_label_one(image)

        return image, label

    def _getLabelByImageName(self, image_name):
        try:
            idx = self.image_names.index(image_name)
            return self.labels[idx]
        except ValueError:
            print(f"Image '{image_name}' not found in the dataset.")
            return None
