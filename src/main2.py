import torch
import torch.nn as nn
import os
import sys
import os.path as osp
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, roc_auc_score
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

batch_size = 32
learning_rate = 0.0001
num_epochs = 5
thresh_hold = 0.5                                                                                                                                                              
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from imblearn.over_sampling import SMOTE
import numpy as np
from torch.utils.data import TensorDataset


def apply_smote(train_dataset):
    # Step 1: Convert image tensors and labels to numpy
    features = []
    labels = []

    for img, label in train_dataset:
        features.append(img.reshape(-1).numpy())
        labels.append(label)  # <-- fix here

    features = np.array(features)
    labels = np.array(labels)

    # Step 2: Apply SMOTE
    sm = SMOTE()
    features_res, labels_res = sm.fit_resample(features, labels)

    # Step 3: Convert back to tensors
    features_res = torch.tensor(features_res, dtype=torch.float32).view(-1, 3, 256, 256)
    labels_res = torch.tensor(labels_res, dtype=torch.float32).view(-1, 1)

    # Step 4: Wrap in TensorDataset
    return TensorDataset(features_res, labels_res)



class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform_label0=None, transform_label1=None):
        self.base_dataset = base_dataset
        self.transform_label0 = transform_label0
        self.transform_label1 = transform_label1

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert PIL to NumPy if using albumentations
        image_np = np.array(image)
        
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

    # full_dataset = ImagesDataset(data_folder, data_csv, transform_for_label_0, transform_for_label_1)

    # random.seed(42)

    # # # Assume full_dataset is already created
    # indices = random.sample(range(len(full_dataset)), 1000)
    # full_dataset = Subset(full_dataset, indices)

    # Split: 80% train, 20% validation
    # Split: 70% train, 15% validation, 15% test
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size  # ensures total consistency

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_dataset = TransformedDataset(train_dataset, transform_for_label_0, transform_for_label_1)

    print("🔁 Applying SMOTE...")
    train_dataset = apply_smote(train_dataset)
    
    val_dataset = TransformedDataset(val_dataset, transform_for_label_0, transform_for_label_0)
    test_dataset = TransformedDataset(test_dataset, transform_for_label_0, transform_for_label_0)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = DenseNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss(alpha=torch.tensor([1.0, 3.0]), gamma=2.0)


    print(f"Learning Rate      : {learning_rate}")
    print(f"Batch Size   : {batch_size}")
    print(f"Epochs : {num_epochs}")
    print(f"Threashold : {thresh_hold}")

    print(f"Total Samples      : {total_size}")
    print(f"Training Samples   : {train_size}")
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
            
            # avg_train_loss = running_loss / len(train_dataloader.dataset)
            # train_losses.append(avg_train_loss)

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
    model_save_path = "/user/HS401/kk01579/APML_PROJECT_KAMALS/saved_models/densenet121_samplying_0.0001.pth"
    # model_save_path = "/home/kamal/APML_PROJECT/saved_models/densenet121_samplying.pth"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    testing_started = datetime.now()
    print(f"Testing started at: {testing_started.strftime('%Y-%m-%d %H:%M:%S')}")

    evaluate(model, test_dataloader, criterion, name="Test")


if __name__ == "__main__":
    init()
