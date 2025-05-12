import os
import sys
import os.path as osp
from datetime import datetime

import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Custom modules
from dataset_processing.images_dataset import ImagesDataset
from dataset_processing.focal_loss import FocalLoss
from dataset_processing.smote import apply_smote_and_concat
from model.densenet import DenseNet
from utils.loggers import Logger
from utils.helper import compute_class_weights
from utils.graphs import plot_confusion_matrix, plot_training_metrics
from dataset_processing.transform_dataset import TransformedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh_hold = 0.5

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

def init(args):
    log_name = "Training_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_data_sampling"
    sys.stdout = Logger(osp.join("./logs", log_name))

    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    f1_scores = []
    recall_scores = []
    train_losses = []
    validation_losses = []
    validation_f1s = []

    data_csv = args.csv
    data_folder = args.img_dir

    transform_for_label_0 = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
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

    full_dataset = ImagesDataset(data_folder, data_csv)

    total_size = len(full_dataset)
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_dataset = TransformedDataset(train_dataset, transform_for_label_0, transform_for_label_1)
    val_dataset = TransformedDataset(val_dataset, transform_for_label_0, transform_for_label_0)
    test_dataset = TransformedDataset(test_dataset, transform_for_label_0, transform_for_label_0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    model = DenseNet(pretrained=True).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = compute_class_weights(train_dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print(f"Learning Rate      : {args.lr}")
    print(f"Batch Size         : {args.batch}")
    print(f"Epochs             : {args.epochs}")
    print(f"Threshold          : {thresh_hold}")
    print(f"Total Samples      : {len(full_dataset)}")
    print(f"Training Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {val_size}")
    print(f"Testing Samples    : {test_size}")

    for epoch in range(args.epochs):
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
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        epoch_loss = running_loss / len(train_dataloader)

        train_losses.append(epoch_loss)
        f1_scores.append(f1)
        recall_scores.append(recall)

        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"Train Loss   : {epoch_loss:.4f}")
        print(f"Train F1     : {f1:.4f}")
        print(f"Train Recall : {recall:.4f}")
        print(f"Train AUC    : {auc:.4f}\n")

        validationLoss, val_f1 = evaluate(model, val_dataloader, criterion, name="Validation")
        validation_losses.append(validationLoss)
        validation_f1s.append(val_f1)

    end_time = datetime.now()
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    plot_training_metrics(
        train_losses, f1_scores, recall_scores,
        validation_losses, validation_f1s,
        filename=log_name,
        learning_rate=args.lr,
        batch_size=args.batch
    )

    model_save_path = "./saved_models/densenet121_sample.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    print(f"Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    evaluate(model, test_dataloader, criterion, name="Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet with custom arguments")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image folder")

    args = parser.parse_args()
    init(args)
