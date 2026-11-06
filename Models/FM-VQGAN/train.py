import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_fscore_support
)
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from models.rgsnet import RGSNet

from fmvqgan_car import train_conditional_fmvqgan, generate_samples_by_class


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


def get_minority_samples(test_dataset, minority_label=1):
    minority_samples, minority_labels = [], []
    for i in range(len(test_dataset)):
        sample, label = test_dataset[i]
        if label == minority_label:
            minority_samples.append(sample)
            minority_labels.append(label)
    return TensorDataset(torch.stack(minority_samples), torch.tensor(minority_labels))


def fgsm_attack(model, images, labels, epsilon):
    if epsilon == 0:
        return images.detach()
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    outputs = model(perturbed_images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        grad_sign = perturbed_images.grad.sign()
        perturbed_images = perturbed_images + epsilon * grad_sign
        perturbed_images = torch.clamp(perturbed_images, -5.0, 5.0)
    return perturbed_images.detach()


def evaluate_fgsm_robustness(model, test_dataset, device, epsilon_values):
    model.eval()
    results = {}
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for epsilon in epsilon_values:
        all_preds, all_targets = [], []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            perturbed_images = fgsm_attack(model, images, labels, epsilon)
            with torch.no_grad():
                outputs = model(perturbed_images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        bin_targets = (np.array(all_targets) != 0).astype(int)
        bin_preds   = (np.array(all_preds)  != 0).astype(int)
        precision = precision_score(bin_targets, bin_preds, pos_label=1, zero_division=0)
        recall    = recall_score(bin_targets, bin_preds, pos_label=1, zero_division=0)
        f1        = f1_score(bin_targets, bin_preds, pos_label=1, zero_division=0)
        results[epsilon] = {'precision': precision, 'recall': recall, 'f1': f1}
    return results


def plot_metrics_decay_curves(results, save_dir='metrics_decay_curves'):
    epsilon_values = sorted(results.keys())
    f1_values        = [results[eps]['f1']        for eps in epsilon_values]
    precision_values = [results[eps]['precision'] for eps in epsilon_values]
    recall_values    = [results[eps]['recall']    for eps in epsilon_values]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for name, values, marker, color in [
        ('precision', precision_values, 's-', 'blue'),
        ('recall',    recall_values,    '^-', 'green'),
        ('f1',        f1_values,        'o-', 'red'),
    ]:
        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_values, values, marker, color=color, linewidth=2)
        plt.xlabel('Epsilon (ε)', fontsize=12)
        plt.ylabel(name.capitalize(), fontsize=12)
        plt.title(f'{name.capitalize()} Decay with FGSM Perturbation', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{name}_decay_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_augment_and_pack_dataset(csv_path, device, train_ratio=0.8):
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
    X_test_norm  = scaler.transform(X_test_raw)

    augment_plan = {
        'Fuzzy': 20000,
        'DoS':   10000,
        'gear':  15000,
        'RPM':   20000,
    }
    minority_classes = [
        int(label_encoder.transform([name])[0])
        for name in augment_plan.keys()
        if name in label_encoder.classes_
    ]
    num_classes = len(label_encoder.classes_)

    full_data_tensor   = torch.tensor(X_train_norm, dtype=torch.float32)
    full_labels_tensor = torch.tensor(y_train_raw, dtype=torch.long)

    G, D, C = train_conditional_fmvqgan(
        full_data_tensor   = full_data_tensor,
        full_labels_tensor = full_labels_tensor,
        num_classes        = num_classes,
        minority_classes   = minority_classes,
        epochs             = 300,
        batch_size         = 256,
        device             = device,
        lambda_gp          = 10,
        lambda_adv         = 0.1,
        lambda_fm          = 0.01,
        lambda_cls         = 0.5,
        n_critic           = 3,
    )

    fake_X_list, fake_y_list = [], []
    for attack_name, count in augment_plan.items():
        if attack_name not in label_encoder.classes_:
            continue
        target_label = int(label_encoder.transform([attack_name])[0])
        fake_arr = generate_samples_by_class(
            G=G,
            target_class=target_label,
            num_samples=count,
            num_classes=num_classes,
            device=device,
        )
        fake_X_list.append(fake_arr)
        fake_y_list.append(np.full((count,), target_label))

    if fake_X_list:
        X_train_augmented = np.vstack([X_train_norm] + fake_X_list)
        y_train_augmented = np.concatenate([y_train_raw] + fake_y_list)
    else:
        X_train_augmented, y_train_augmented = X_train_norm, y_train_raw

    train_dataset = create_sliding_window_dataset(
        X_train_augmented, y_train_augmented,
        seq_len=27, step=5, normal_label=normal_label
    )
    test_dataset = create_sliding_window_dataset(
        X_test_norm, y_test_raw,
        seq_len=27, step=5, normal_label=normal_label
    )

    return train_dataset, test_dataset, label_encoder


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    EPOCHS = 300
    LR = 0.001

    CSV_PATH   = "data/car_dataset.csv"
    CACHE_PATH = "data/augmented_cache_car_cond.pth"

    if os.path.exists(CACHE_PATH):
        checkpoint = torch.load(CACHE_PATH, weights_only=False)
        train_dataset = checkpoint['train_dataset']
        test_dataset  = checkpoint['test_dataset']
        label_encoder = checkpoint['label_encoder']
    else:
        train_dataset, test_dataset, label_encoder = load_augment_and_pack_dataset(
            CSV_PATH, device, train_ratio=0.8
        )
        torch.save({
            'train_dataset': train_dataset,
            'test_dataset':  test_dataset,
            'label_encoder': label_encoder
        }, CACHE_PATH)

    num_classes = len(label_encoder.classes_)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model     = RGSNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
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
    all_preds   = np.array(all_preds)
    all_indices = np.arange(len(label_encoder.classes_))

    report_str = classification_report(
        all_targets, all_preds,
        labels=all_indices,
        target_names=label_encoder.classes_,
        digits=4, zero_division=0
    )
    split_idx = report_str.find('\n    accuracy')

    bin_targets = (all_targets != 0).astype(int)
    bin_preds   = (all_preds   != 0).astype(int)
    bin_metrics = precision_recall_fscore_support(bin_targets, bin_preds, labels=[0, 1])

    torch.save(model.state_dict(), 'hybrid_cnn_model.pth')

    epsilon_values = np.linspace(0, 0.1, 11)
    results = evaluate_fgsm_robustness(model, test_dataset, device, epsilon_values)
    plot_metrics_decay_curves(results)


if __name__ == "__main__":
    main()
