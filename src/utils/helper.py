from collections import Counter
import torch
from torch.utils.data import (
    DataLoader, Dataset, Subset, ConcatDataset, random_split, TensorDataset
)

import os
import sys
import os.path as osp
from datetime import datetime

import random
import numpy as np
import cv2  # Needed for cv2.BORDER_REFLECT
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, Dataset, Subset, ConcatDataset, random_split, TensorDataset
)


from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from dataset_processing.custom_dataset import CustomDataset
from imblearn.over_sampling import SMOTE
from utils.graphs import plot_training_metrics

# Custom modules
from dataset_processing.images_dataset import ImagesDataset
from dataset_processing.focal_loss import FocalLoss
from dataset_processing.smote import apply_smote_and_concat
from model.densenet import DenseNet
from utils.loggers import Logger
from utils.helper import compute_class_weights
from utils.graphs import plot_confusion_matrix
from utils.graphs import plot_training_metrics

def compute_class_weights(dataset):
    targets = []
    for _, label in dataset:
        targets.append(int(label)) 
    
    counter = Counter(targets)
    count_0 = counter[0]
    count_1 = counter[1]

    pos_weight = torch.tensor([count_0 / count_1], dtype=torch.float)
    return pos_weight


def count_labels_multiple_datasets(*datasets):
    count_0 = 0
    count_1 = 0

    for dataset in datasets:
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        if isinstance(dataset, ConcatDataset):
            sub_datasets = dataset.datasets
        else:
            sub_datasets = [dataset]

        for ds in sub_datasets:
            for i in range(len(ds)):
                sample = ds[i]
                label = sample[1] if isinstance(sample, tuple) else sample['label']

                if label == 0:
                    count_0 += 1
                elif label == 1:
                    count_1 += 1

    print(f"Class 0 count: {count_0}")
    print(f"Class 1 count: {count_1}")

    return count_0, count_1




def duplicate_and_augment_dataset(dataset, label=1):

    original_data = []
    augmented_data = []

    augmentations = [
        A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), A. ToTensorV2()]),
        A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()]),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), ToTensorV2()]),
        A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), A.Resize(256, 256), A. ToTensorV2()]),
    ]

    to_tensor_transform = A.Compose([ToTensorV2()])

    print(f"[INFO] Starting augmentation. Total original samples: {len(dataset)}")

    for idx, (image, target_label) in enumerate(dataset):
        # Convert image to numpy
        if isinstance(image, np.ndarray):
            image_np = image
        elif isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()  # Convert CxHxW to HxWxC
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Apply ToTensorV2 for original
        original_image = to_tensor_transform(image=image_np)["image"]
        original_data.append((original_image, target_label))

        # Apply different augmentations for specified label
        if target_label == label:
            for aug in augmentations:
                augmented_image = aug(image=image_np)["image"]
                augmented_data.append((augmented_image, target_label))

    # Combine and create dataset
    final_data = original_data + augmented_data
    final_dataset = CustomDataset(final_data, transform=to_tensor_transform)

    return final_dataset

