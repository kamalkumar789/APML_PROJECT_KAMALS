import torch
import torch.nn as nn
import os
import sys
import os.path as osp
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix
from dataset_processing.images_dataset import ImagesDataset
from model.densenet import DenseNet
from datetime import datetime
from torch.utils.data import random_split
from utils.loggers import Logger
from utils.helper import compute_class_weights
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # Needed for cv2.BORDER_REFLECT
import numpy as np
from dataset_processing.focal_loss import FocalLoss
import seaborn as sns
from torch.utils.data import ConcatDataset, TensorDataset
from imblearn.over_sampling import SMOTE
import torch
import numpy as np
from sklearn.utils import shuffle
from dataset_processing.smote import apply_smote_and_concat

batch_size = 32
learning_rate = 0.0001
num_epochs = 100
thresh_hold = 0.5                                                                                                                                                              
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch.utils.data import ConcatDataset, TensorDataset
from torchvision.transforms import ToTensor

# def augment_label_1(dataset, transform, times=3):
#     augmented_data = []
#     to_tensor = ToTensor()  # Convert PIL Image to Tensor
    
#     for img, label in dataset:
#         if label == 1:
#             for _ in range(times):
#                 # Convert the image to a tensor
#                 img_tensor = to_tensor(img)
#                 augmented_data.append((img_tensor, 1))
    
#     # Wrap into a dataset
#     augmented_dataset = TensorDataset(
#         torch.stack([x[0] for x in augmented_data]),
#         torch.tensor([x[1] for x in augmented_data])
#     )

#     # Combine with the original dataset
#     return ConcatDataset([dataset, augmented_dataset])


from torch.utils.data import Subset, ConcatDataset

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
    plt.savefig(f'{name.lower()}_confusion_matrix.png')
    print(f"[INFO] Confusion matrix saved to: {name.lower()}_confusion_matrix.png")
    plt.close()

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform_label0=None, transform_label1=None):
        self.base_dataset = base_dataset
        self.transform_label0 = transform_label0
        self.transform_label1 = transform_label1

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert PIL to NumPy if using albumentations
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.asarray(image)  # Allows a copy if needed # Avoid specifying dtype explicitly        

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
def plot_training_metrics(train_losses, f1_scores, recall_scores, validation_losses, filename):
    min_len = min(len(train_losses), len(validation_losses), len(f1_scores), len(recall_scores))
    epochs = range(1, min_len + 1)

    plt.figure(figsize=(12, 5))

    # Plot Traininmalignantidation Loss on same plot
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
    plt.plot(epochs, f1_scores[:min_len], label='F1 Score', marker='o', color='blue')
    plt.plot(epochs, recall_scores[:min_len], label='Recall', marker='x', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1 Score and Recall per Epoch')
    plt.legend()
    plt.grid(True)

    # Add annotations for learning rate and batch size
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

    plot_confusion_matrix(all_labels, all_preds, name)

    return val_loss


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
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
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
    # indices = random.sample(range(len(full_dataset)), 10000)
    # full_dataset = Subset(full_dataset, indices)
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size  # ensures total consistency

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(len(train_dataset))
    train_dataset = apply_smote_and_concat(train_dataset, image_shape=(256, 256, 3), k_neighbors=3)
    count_labels_multiple_datasets(train_dataset)

    print(len(train_dataset))

    train_dataset = TransformedDataset(train_dataset, transform_for_label_0, transform_for_label_1)
    val_dataset = TransformedDataset(val_dataset, transform_for_label_0, transform_for_label_0)
    test_dataset = TransformedDataset(test_dataset, transform_for_label_0, transform_for_label_0)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = DenseNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss(alpha=torch.tensor([1.0, 3.0]), gamma=1.0)


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

        validationLoss = evaluate(model, val_dataloader, criterion, name="Validation")
        validation_losses.append(validationLoss)
    
    end_time = datetime.now()
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    plot_training_metrics(train_losses, f1_scores, recall_scores, validation_losses, filename=log_name)

    # ✅ Save the trained model
    model_save_path = "/user/HS401/kk01579/APML_PROJECT_KAMALS/saved_models/densenet121_samplying_smote.pth"
    # model_save_path = "/home/kamal/APML_PROJECT/saved_models/densenet121_samplying.pth"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    testing_started = datetime.now()
    print(f"Testing started at: {testing_started.strftime('%Y-%m-%d %H:%M:%S')}")

    evaluate(model, test_dataloader, criterion, name="Test")


if __name__ == "__main__":
    init()
