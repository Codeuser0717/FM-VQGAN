import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from models.rgsnet import RGSNet

def create_sliding_window_dataset(X, y, seq_len=27, step=5, normal_label=0):
    image_list = []
    label_list = []
    for i in range(0, len(X) - seq_len, step):
        chunk_X = X[i : i + seq_len]
        chunk_y = y[i : i + seq_len]
        img = chunk_X.reshape(3, 9, -1)
        counts = np.bincount(chunk_y)
        counts_attack = counts.copy()
        if normal_label < len(counts_attack):
            counts_attack[normal_label] = 0
        if counts_attack.sum() >= 3:
            final_label = counts_attack.argmax()
        else:
            final_label = normal_label
        image_list.append(img)
        label_list.append(final_label)
    X_images = torch.tensor(np.array(image_list), dtype=torch.float32)
    y_image_labels = torch.tensor(np.array(label_list), dtype=torch.long)
    return TensorDataset(X_images, y_image_labels)

def load_and_pack_dataset(csv_path, device, train_ratio=0.8):
    df = pd.read_csv(csv_path)

    label_col = df.columns[-1]
    y_raw = df[label_col].values
    X_raw = df.drop(columns=[label_col]).values

    label_encoder = LabelEncoder()
    custom_classes = ['R', 'DoS', 'Fuzzy', 'gear', 'RPM']
    label_encoder.classes_ = np.array(custom_classes)
    y_encoded = label_encoder.transform(y_raw)

    normal_label = 0

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_encoded,
        train_size=train_ratio,
        stratify=y_encoded,
        random_state=42
    )

    scaler = QuantileTransformer(output_distribution='normal')
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_test_norm = scaler.transform(X_test_raw)

    train_dataset = create_sliding_window_dataset(
        X_train_norm, y_train_raw, seq_len=27, step=5, normal_label=normal_label
    )
    test_dataset = create_sliding_window_dataset(
        X_test_norm, y_test_raw, seq_len=27, step=5, normal_label=normal_label
    )

    return train_dataset, test_dataset, label_encoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    EPOCHS = 300
    LR = 0.001

    CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'car_dataset.csv')
    CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data', 'cnn_cache.pth')

    if os.path.exists(CACHE_PATH):
        checkpoint = torch.load(CACHE_PATH, weights_only=False)
        train_dataset = checkpoint['train_dataset']
        test_dataset = checkpoint['test_dataset']
        label_encoder = checkpoint['label_encoder']
    else:
        train_dataset, test_dataset, label_encoder = load_and_pack_dataset(CSV_PATH, device, train_ratio=0.8)
        torch.save({
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'label_encoder': label_encoder
        }, CACHE_PATH)

    num_classes = len(label_encoder.classes_)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RGSNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    all_indices = np.arange(len(label_encoder.classes_))

    report_str = classification_report(
        all_targets,
        all_preds,
        labels=all_indices,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )

    split_idx = report_str.find('\n    accuracy')

    bin_targets = (all_targets != 0).astype(int)
    bin_preds = (all_preds != 0).astype(int)
    bin_metrics = precision_recall_fscore_support(bin_targets, bin_preds, labels=[0, 1])

if __name__ == "__main__":
    main()
