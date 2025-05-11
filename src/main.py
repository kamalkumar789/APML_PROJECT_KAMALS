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
import matplotlib

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, Dataset, Subset, ConcatDataset, random_split, TensorDataset
)
from torchvision import transforms
from torchvision.transforms import ToTensor

from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

# Custom modules
from dataset_processing.images_dataset import ImagesDataset
from dataset_processing.focal_loss import FocalLoss
from dataset_processing.smote import apply_smote_and_concat
from model.densenet import DenseNet
from utils.loggers import Logger
from utils.helper import compute_class_weights



batch_size = 32
learning_rate = 0.0001
num_epochs = 10
thresh_hold = 0.5                                                                                                                                                              
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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



def duplicate_and_augment_dataset(dataset, label=1):

    original_data = []
    augmented_data = []

    # Define 5 different augmentation pipelines
    augmentations = [
        A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), A. ToTensorV2()]),
        A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()]),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), ToTensorV2()]),
        A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), A.Resize(256, 256), A. ToTensorV2()]),
    ]

    # Simple tensor conversion (for original images)
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


def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'/user/HS401/kk01579/APML_PROJECT_KAMALS/src/visuals/{name.lower()}_confusion_matrix_classweights_main_1.png')
    print(f"[INFO] Confusion matrix saved to: {name.lower()}_confusion_matrix_main_1.png")
    plt.close()

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

# 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
def plot_training_metrics(train_losses, f1_scores, recall_scores, validation_losses, validation_f1_scores, filename):
    min_len = min(len(train_losses), len(validation_losses), len(f1_scores), len(recall_scores), len(validation_f1_scores))
    epochs = range(1, min_len + 1)

    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses[:min_len], label='Train Loss', marker='o', color='red')
    plt.plot(epochs, validation_losses[:min_len], label='Validation Loss', marker='x', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Plot F1 Score and Recall
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores[:min_len], label='Train F1 Score', marker='o', color='blue')
    plt.plot(epochs, validation_f1_scores[:min_len], label='Validation F1 Score', marker='s', color='purple')
    plt.plot(epochs, recall_scores[:min_len], label='Train Recall', marker='x', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1 Score and Recall per Epoch')
    plt.legend()
    plt.grid(True)

    # Annotate Learning Rate and Batch Size
    plt.figtext(0.15, 0.05, f'Learning Rate: {learning_rate}', fontsize=10, color='black')
    plt.figtext(0.15, 0.01, f'Batch Size: {batch_size}', fontsize=10, color='black')

    plt.tight_layout()
    save_path = filename + ".png"
    plt.savefig(save_path)
    print(f"[INFO] Training plot saved to: {save_path}")

    if matplotlib.is_interactive():
        plt.show()


def evaluate(model, dataloader, criterion, name="Validation"):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > thresh_hold).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    val_loss = running_loss / len(dataloader)
    val_f1 = f1_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds)
    val_auc = roc_auc_score(all_labels, all_probs)

    print(f"\n{name} Evaluation:")
    print(f"{name} Loss   : {val_loss:.4f}")
    print(f"{name} F1     : {val_f1:.4f}")
    print(f"{name} Recall : {val_recall:.4f}")
    print(f"{name} AUC    : {val_auc:.4f}")

    if name == "Test":
        plot_confusion_matrix(all_labels, all_preds, name)

    return val_loss, val_f1


def init():
    log_name = "Training_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_data_sampling"
    sys.stdout = Logger(osp.join("/user/HS401/kk01579/APML_PROJECT_KAMALS/src/logs", log_name))
    # sys.stdout = Logger(osp.join("/home/kamal/AppliedMachineLearning/APML_PROJECT_KAMALS/src/logs", log_name))

    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    f1_scores = []
    recall_scores = []
    train_losses = []
    validation_losses = []
    validation_f1s = []


    # Path to CSV and folder
    data_csv = "/user/HS401/kk01579/archive/train.csv"
    data_folder = "/user/HS401/kk01579/archive/train"

    # data_csv = "/home/kamal/archive/train.csv"
    # data_folder = "/home/kamal/archive/train"

    transform_for_label_0 = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Add normalization
        ToTensorV2()  # This converts to FloatTensor
    ])
 
    transform_for_label_1 = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ], p=1.0),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]) 

    print("\nTransform for label 0:\n", transform_for_label_0)
    print("\nTransform for label 1:\n", transform_for_label_1)

    full_dataset = ImagesDataset(data_folder, data_csv, None, None)
    # random.seed(42)

    # # # Assume full_dataset is already created
    # indices = random.sample(range(len(full_dataset)), 1000)
    # full_dataset = Subset(full_dataset, indices)
    
    

    # full_dataset = duplicate_and_augment_dataset(full_dataset, label=1)
    total_size = len(full_dataset)
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size  # ensures total consistency

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_dataset = TransformedDataset(train_dataset, transform_for_label_0, transform_for_label_0)    
    val_dataset = TransformedDataset(val_dataset, transform_for_label_0, transform_for_label_0)
    test_dataset = TransformedDataset(test_dataset, transform_for_label_0, transform_for_label_0)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = DenseNet(pretrained=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = compute_class_weights(train_dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print(f"Learning Rate      : {learning_rate}")
    print(f"Batch Size   : {batch_size}")
    print(f"Epochs : {num_epochs}")
    print(f"Threashold : {thresh_hold}")

    print(f"Total Samples      : {len(train_dataset) + val_size + test_size}")
    print(f"Training Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {val_size}")
    print(f"Testing Samples : {test_size}")
    


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.float().to(device)

            labels = labels.view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > thresh_hold).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        
        # Training Metrics
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)        
        auc = roc_auc_score(all_labels, all_probs)
        epoch_loss = running_loss / len(train_dataloader)
        train_losses.append(epoch_loss) 
        f1_scores.append(f1)
        recall_scores.append(recall)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Train Loss   : {epoch_loss:.4f}")
        print(f"Train F1     : {f1:.4f}")
        print(f"Train Recall : {recall:.4f}")
        print(f"Train AUC    : {auc:.4f}\n")

        validationLoss, val_f1 = evaluate(model, val_dataloader, criterion, name="Validation")
        validation_losses.append(validationLoss)
        validation_f1s.append(val_f1)

    
    end_time = datetime.now()
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    plot_training_metrics(train_losses, f1_scores, recall_scores, validation_losses, validation_f1s,  filename=log_name)

    # ✅ Save the trained model
    model_save_path = "/user/HS401/kk01579/APML_PROJECT_KAMALS/saved_models/densenet121_samplying_50_702010_new.pth"
    # model_save_path = "/home/kamal/APML_PROJECT/saved_models/densenet121_samplying.pth"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    
    print(f"Model saved at: {model_save_path}")

    testing_started = datetime.now()
    print(f"Testing started at: {testing_started.strftime('%Y-%m-%d %H:%M:%S')}")

    evaluate(model, test_dataloader, criterion, name="Test")


if __name__ == "__main__":
    init()
