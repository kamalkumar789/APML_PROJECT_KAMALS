from imblearn.over_sampling import SMOTE
from torch.utils.data import ConcatDataset, TensorDataset
import torch
import numpy as np


def apply_smote_and_concat(train_dataset, image_shape=(256, 256, 3), k_neighbors=3):
    """
    Applies SMOTE to balance the dataset and concatenates the augmented data with the original train dataset.
    """
    train_images = []
    train_labels = []

    for img, label in train_dataset:
        train_images.append(np.array(img))
        train_labels.append(label)
    
    # Flatten the images to 2D (if needed, depending on your model input)
    train_images_flat = [img.flatten() for img in train_images]
    train_images_flat = np.array(train_images_flat)
    train_labels = np.array(train_labels)

    smote = SMOTE(sampling_strategy=0.2, random_state=30)
    train_images_resampled, train_labels_resampled = smote.fit_resample(train_images_flat, train_labels)
    # Reshape the images back to original size (assuming 256x256x3)
    train_images_resampled = [img.reshape(256, 256, 3) for img in train_images_resampled]

    # Create a new dataset with the resampled data
    augmented_train_dataset = [(torch.tensor(img), label) for img, label in zip(train_images_resampled, train_labels_resampled)]
    
    combined_dataset = ConcatDataset([train_dataset, augmented_train_dataset])

    return combined_dataset
