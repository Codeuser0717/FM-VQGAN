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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.n_e = num_embeddings
        self.e_dim = embedding_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, x):
        flat_x = x.view(-1, self.e_dim)
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(x.shape)

        quantized = x + (quantized - x).detach()
        return quantized, encoding_indices

class VQModel(nn.Module):
    def __init__(self, input_dim=78, latent_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, indices

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    feature_columns = [col for col in df.columns if col != 'Label']
    X = df[feature_columns].values
    y = df['Label'].values

    scaler = QuantileTransformer(output_distribution='normal')
    X_norm = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_encoded,
        train_size=0.8,
        stratify=y_encoded,
        random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    return X_train_tensor, X_test_tensor, scaler, label_encoder

def vqvae_loss(recon_x, x, z_e, z_q, commitment_cost=0.25):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    vq_loss = torch.mean((z_e.detach() - z_q)**2)
    commit_loss = torch.mean((z_e - z_q.detach())**2)
    return recon_loss + vq_loss + commitment_cost * commit_loss

def train_vqvae(model, train_data, epochs=300, batch_size=128, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(train_data, train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x, _ = batch
            x = x.to(device)

            optimizer.zero_grad()
            recon_x, z_e, z_q, indices = model(x)
            loss = vqvae_loss(recon_x, x, z_e, z_q)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)

    return model

def generate_fake_data(generator, num_samples, latent_dim=64, num_embeddings=128, device='cuda'):
    generator.eval()
    with torch.no_grad():
        indices = torch.randint(0, num_embeddings, (num_samples,)).to(device)
        z_q = generator.quantizer.embedding(indices)
        fake_data = generator.decoder(z_q)
    return fake_data.cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    EPOCHS = 300
    LR = 0.001

    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'car_dataset.csv')
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vqvae_model.pth')
    CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data', 'augmented_cache_vqvae.pth')

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
        train_size=0.8,
        stratify=y_encoded,
        random_state=42
    )

    scaler = QuantileTransformer(output_distribution='normal')
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_test_norm = scaler.transform(X_test_raw)

    augment_plan = {
        'Fuzzy': 20000,
        'DoS': 10000,
        'gear': 15000,
        'RPM': 20000
    }

    fake_X_list = []
    fake_y_list = []

    input_dim = X_train_norm.shape[1]
    latent_dim = 64
    num_embeddings = 128
    vqvae_model = VQModel(input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings)
    trained_model = train_vqvae(vqvae_model, torch.tensor(X_train_norm, dtype=torch.float32), epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    torch.save(trained_model.state_dict(), model_save_path)

    trained_model.to(device)

    for attack_name, count in augment_plan.items():
        if attack_name not in label_encoder.classes_:
            continue

        target_label = label_encoder.transform([attack_name])[0]

        real_data = X_train_norm[y_train_raw == target_label]
        if len(real_data) == 0:
            continue

        fake_data = generate_fake_data(trained_model, count, latent_dim=latent_dim, num_embeddings=num_embeddings, device=device)

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

    plot_tsne(original_sample, generated_sample, "VQVAE")

    train_dataset = create_sliding_window_dataset(
        X_train_augmented, y_train_augmented, seq_len=27, step=5, normal_label=normal_label
    )
    test_dataset = create_sliding_window_dataset(
        X_test_norm, y_test_raw, seq_len=27, step=5, normal_label=normal_label
    )

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
