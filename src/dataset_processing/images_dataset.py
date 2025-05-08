import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class ImagesDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform_for_label_0=None, transform_for_label_1=None, duplication_factor=3):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)

        self.image_names = self.data['image_name'].tolist()
        self.labels = self.data['target'].tolist()
        self.ages = self.data['age_approx'].tolist()
        self.sex = self.data['sex']
        self.patientsIds = self.data['patient_id']

        self.transform_label_zero = transform_for_label_0
        self.transform_label_one = transform_for_label_1

        # Duplication factor for label 1
        self.duplication_factor = duplication_factor

        # Create a new dataset with duplicated entries for label 1
        self.augmented_image_names = self.image_names.copy()
        self.augmented_labels = self.labels.copy()

        # Duplicate images with label 1 and apply the transformation if applicable
        for idx in range(len(self.image_names)):
            if self.labels[idx] == 1:
                for _ in range(self.duplication_factor - 1):  # Subtract 1 because the original image is already included
                    # Add the original image
                    self.augmented_image_names.append(self.image_names[idx])
                    self.augmented_labels.append(self.labels[idx])

                    # Apply the transformation for label 1
                    image_path = os.path.join(self.image_folder, self.image_names[idx])
                    image = Image.open(image_path + ".jpg").convert('RGB')

                    # Convert PIL image to NumPy array
                    image = np.array(image)

                    # Apply the transformation for label 1
                    if self.transform_label_one:
                        augmented_img = self.transform_label_one(image=image)['image']

                        # Store augmented image and label
                        self.augmented_image_names.append(self.image_names[idx])  # Append the same name or modified name if needed
                        self.augmented_labels.append(1)  # Assign label 1 for augmented images

    def __len__(self):
        return len(self.augmented_image_names)

    def __getitem__(self, idx):
        image_name = self.augmented_image_names[idx]
        label = self.augmented_labels[idx]

        image_path = os.path.join(self.image_folder, image_name)
        if not image_path.lower().endswith('.jpg'):
            image_path += ".jpg"

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        return image, label

    def _getLabelByImageName(self, image_name):
        try:
            idx = self.image_names.index(image_name)
            return self.labels[idx]
        except ValueError:
            print(f"Image '{image_name}' not found in the dataset.")
            return None
