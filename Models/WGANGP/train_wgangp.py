import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from models.rgsnet import RGSNet

def plot_tsne(original_data, generated_data, model_name):
    all_data = np.vstack([original_data, generated_data])
    all_labels = np.zeros(len(all_data))
    all_labels[len(original_data):] = 1
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:len(original_data), 0], tsne_result[:len(original_data), 1],
                c='blue', label='Original Data', alpha=0.5, s=10)
    plt.scatter(tsne_result[len(original_data):, 0], tsne_result[len(original_data):, 1],
                c='red', label='Generated Data', alpha=0.5, s=10)
    plt.title(f't-SNE Visualization of {model_name} Generated vs Original Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    save_path = f"{model_name}/tsne_{model_name.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_wgangp(real_data, epochs=300, batch_size=128, lr=1e-4, latent_dim=128, device='cuda'):
    input_dim = real_data.shape[1]

    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    real_data = real_data.to(device)
    data_loader = DataLoader(TensorDataset(real_data), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            real_samples = batch[0]
            batch_size_actual = real_samples.shape[0]

            optimizer_d.zero_grad()

            real_output = discriminator(real_samples)

            noise = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_samples = generator(noise)
            fake_output = discriminator(fake_samples.detach())

            alpha = torch.rand(batch_size_actual, 1).to(device)
            interpolated = alpha * real_samples + (1 - alpha) * fake_samples.detach()
            interpolated.requires_grad_(True)
            interpolated_output = discriminator(interpolated)

            gradients = torch.autograd.grad(
                outputs=interpolated_output,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interpolated_output),
                create_graph=True,
                retain_graph=True
            )[0]

            gradients = gradients.view(batch_size_actual, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            d_loss = -torch.mean(real_output) + torch.mean(fake_output) + 10 * gradient_penalty
            d_loss.backward()
            optimizer_d.step()

            if i % 5 == 0:
                optimizer_g.zero_grad()
                noise = torch.randn(batch_size_actual, latent_dim).to(device)
                fake_samples = generator(noise)
                fake_output = discriminator(fake_samples)
                g_loss = -torch.mean(fake_output)
                g_loss.backward()
                optimizer_g.step()


    return generator

def generate_fake_data(generator, num_samples, latent_dim=128, device='cuda'):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_data = generator(noise)
    return fake_data.cpu().numpy()

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
    X_test_norm = scaler.transform(X_test_raw)


    fake_X_list = []
    fake_y_list = []

    for attack_name, count in augment_plan.items():
        if attack_name not in label_encoder.classes_:
            continue

        target_label = label_encoder.transform([attack_name])[0]

        real_data = X_train_norm[y_train_raw == target_label]
        if len(real_data) == 0:
            continue

        real_tensor = torch.tensor(real_data, dtype=torch.float32)

        generator = train_wgangp(real_tensor, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device)
        generator.eval()

        fake_data = generate_fake_data(generator, count, device=device)

        fake_X_list.append(fake_data)
        fake_y_list.append(np.full((count,), target_label))

    if fake_X_list:
        X_train_augmented = np.vstack([X_train_norm] + fake_X_list)
        y_train_augmented = np.concatenate([y_train_raw] + fake_y_list)
    else:
        X_train_augmented, y_train_augmented = X_train_norm, y_train_raw

    np.random.seed(42)
    sample_size = min(1000, len(X_train_norm))
    original_sample_indices = np.random.choice(len(X_train_norm), sample_size, replace=False)
    original_sample = X_train_norm[original_sample_indices]

    if fake_X_list:
        all_fake_data = np.vstack(fake_X_list)
        if len(all_fake_data) >= sample_size:
            generated_sample_indices = np.random.choice(len(all_fake_data), sample_size, replace=False)
            generated_sample = all_fake_data[generated_sample_indices]
        else:
            generated_sample = all_fake_data
            additional_sample_size = sample_size - len(generated_sample)
            additional_indices = np.random.choice(len(X_train_norm), additional_sample_size, replace=False)
            generated_sample = np.vstack([generated_sample, X_train_norm[additional_indices]])
    else:
        generated_sample = X_train_norm[:sample_size]

    train_dataset = create_sliding_window_dataset(
        X_train_augmented, y_train_augmented, seq_len=27, step=5, normal_label=normal_label
    )
    test_dataset = create_sliding_window_dataset(
        X_test_norm, y_test_raw, seq_len=27, step=5, normal_label=normal_label
    )

    return train_dataset, test_dataset, label_encoder, original_sample, generated_sample

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    EPOCHS = 300
    LR = 0.001

    CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'car_dataset.csv')
    CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data', 'augmented_cache_wgangp.pth')

    if os.path.exists(CACHE_PATH):
        checkpoint = torch.load(CACHE_PATH, weights_only=False)
        train_dataset = checkpoint['train_dataset']
        test_dataset = checkpoint['test_dataset']
        label_encoder = checkpoint['label_encoder']
    else:
        train_dataset, test_dataset, label_encoder, original_sample, generated_sample = load_augment_and_pack_dataset(CSV_PATH, device, train_ratio=0.8)
        plot_tsne(original_sample, generated_sample, "WGANGP")
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
