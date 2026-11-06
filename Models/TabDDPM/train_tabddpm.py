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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from models.rgsnet import RGSNet

def plot_tsne(original_data, generated_data, model_name):
    print(f"\n生成{model_name}的t-SNE可视化图...")
    all_data = np.vstack([original_data, generated_data])
    all_labels = np.zeros(len(all_data))
    all_labels[len(original_data):] = 1
    from sklearn.manifold import TSNE
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
    print(f"t-SNE图已保存到: {save_path}")
    plt.close()

def fgsm_attack(model, images, labels, epsilon):
    if epsilon == 0:
        return images.detach()

    perturbed_images = images.clone().detach().requires_grad_(True)

    outputs = model(perturbed_images)

    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        grad_sign = perturbed_images.grad.sign()
        x_adv = perturbed_images + epsilon * grad_sign
        x_adv = torch.clamp(x_adv, -5.0, 5.0)

    return x_adv.detach()

def evaluate_fgsm_robustness(model, test_dataset, device, epsilon_values):
    model.eval()
    results = {}

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epsilon in epsilon_values:
        all_preds = []
        all_targets = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            perturbed_images = fgsm_attack(model, images, labels, epsilon)

            with torch.no_grad():
                outputs = model(perturbed_images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        bin_targets = (np.array(all_targets) != 0).astype(int)
        bin_preds = (np.array(all_preds) != 0).astype(int)

        precision = precision_score(bin_targets, bin_preds, pos_label=1, zero_division=0)
        recall = recall_score(bin_targets, bin_preds, pos_label=1, zero_division=0)
        f1 = f1_score(bin_targets, bin_preds, pos_label=1, zero_division=0)

        results[epsilon] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"ε={epsilon:.3f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return results

def plot_metrics_decay_curves(results, save_dir='TabDDPM/metrics_decay_curves'):
    epsilon_values = sorted(results.keys())
    f1_values = [results[eps]['f1'] for eps in epsilon_values]
    precision_values = [results[eps]['precision'] for eps in epsilon_values]
    recall_values = [results[eps]['recall'] for eps in epsilon_values]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, precision_values, 's-', color='blue', linewidth=2)
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision Decay with FGSM Perturbation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    precision_save_path = os.path.join(save_dir, 'precision_decay_curve.png')
    plt.savefig(precision_save_path, dpi=300, bbox_inches='tight')
    print(f"Precision衰减曲线已保存为: {precision_save_path}")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, recall_values, '^-', color='green', linewidth=2)
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall Decay with FGSM Perturbation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    recall_save_path = os.path.join(save_dir, 'recall_decay_curve.png')
    plt.savefig(recall_save_path, dpi=300, bbox_inches='tight')
    print(f"Recall衰减曲线已保存为: {recall_save_path}")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, f1_values, 'o-', color='red', linewidth=2)
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Decay with FGSM Perturbation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    f1_save_path = os.path.join(save_dir, 'f1_decay_curve.png')
    plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
    print(f"F1衰减曲线已保存为: {f1_save_path}")
    plt.show()

class TabDDPM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_steps=1000):
        super(TabDDPM, self).__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        alpha_cum_prod = self.alpha_cum_prod.to(x.device)
        t_embed = self.time_embed(t.reshape(-1, 1))
        input = torch.cat([x, t_embed], dim=1)
        return self.net(input)

    def sample(self, num_samples, device='cuda'):
        self.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, self.input_dim).to(device)

            alphas = self.alphas.to(device)
            alpha_cum_prod = self.alpha_cum_prod.to(device)
            betas = self.betas.to(device)

            for i in reversed(range(self.num_steps)):
                t = torch.full((num_samples,), i, dtype=torch.float32).to(device)
                noise_pred = self(x, t)

                alpha = alphas[i]
                alpha_cum_prod_i = alpha_cum_prod[i]
                beta = betas[i]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cum_prod_i)) * noise_pred) + torch.sqrt(beta) * noise

        return x.cpu().numpy()

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

def load_and_preprocess_data(csv_path):
    print("Loading and preprocessing data...")
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

    return X_train_tensor, X_test_tensor, y_train, y_test, scaler, label_encoder

def train_tabddpm(model, train_data, epochs=300, batch_size=128, lr=1e-4, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_data, train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training TabDDPM on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x, _ = batch
            x = x.to(device)

            t = torch.randint(0, model.num_steps, (x.shape[0],), dtype=torch.float32).to(device)

            noise = torch.randn_like(x)

            alpha_cum_prod = model.alpha_cum_prod.to(x.device)[t.long()].reshape(-1, 1)
            x_t = torch.sqrt(alpha_cum_prod) * x + torch.sqrt(1 - alpha_cum_prod) * noise

            noise_pred = model(x_t, t)

            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model

def generate_fake_data(model, num_samples, device='cuda'):
    model.eval()
    with torch.no_grad():
        fake_data = model.sample(num_samples, device)
    return fake_data

def load_augment_and_pack_dataset(csv_path, device, train_ratio=0.8):
    print("\n[阶段 1/4] 正在从 CSV 读取数据，并执行【分层抽样】切分...")
    df = pd.read_csv(csv_path)

    label_col = df.columns[-1]
    y_raw = df[label_col].values
    X_raw = df.drop(columns=[label_col]).values

    label_encoder = LabelEncoder()
    custom_classes = ['R', 'DoS', 'Fuzzy', 'gear', 'RPM']
    label_encoder.classes_ = np.array(custom_classes)
    y_encoded = label_encoder.transform(y_raw)

    normal_label = 0

    print(f"类标映射关系: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_encoded,
        train_size=train_ratio,
        stratify=y_encoded,
        random_state=42
    )

    print("\n[阶段 2/3] 基于训练集分布进行归一化处理...")
    scaler = QuantileTransformer(output_distribution='normal')
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_test_norm = scaler.transform(X_test_raw)

    print(f"\n训练集大小: {len(X_train_norm)}")
    print(f"测试集大小: {len(X_test_norm)}")

    print("\n[阶段 3/4] 训练TabDDPM模型并执行数据增强...")
    input_dim = X_train_norm.shape[1]
    tabddpm_model = TabDDPM(input_dim).to(device)
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    trained_model = train_tabddpm(tabddpm_model, X_train_tensor, epochs=300, batch_size=128, device=device)

    model_save_path = "TabDDPM/tabddpm_model.pth"
    os.makedirs("TabDDPM", exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"TabDDPM模型已保存到: {model_save_path}")

    fake_X_list = []
    fake_y_list = []

    for attack_name, count in augment_plan.items():
        if attack_name not in label_encoder.classes_:
            continue
        target_label = label_encoder.transform([attack_name])[0]
        print(f"\n> 正在用 TabDDPM 生成 [{attack_name}] 样本，目标数量: {count} 条...")

        fake_data = generate_fake_data(trained_model, count, device=device)
        fake_X_list.append(fake_data)
        fake_y_list.append(np.full((count,), target_label))

    if fake_X_list:
        X_train_augmented = np.vstack([X_train_norm] + fake_X_list)
        y_train_augmented = np.concatenate([y_train_raw] + fake_y_list)
    else:
        X_train_augmented, y_train_augmented = X_train_norm, y_train_raw

    print(f"\n训练集增强完毕！原始: {len(X_train_norm)} -> 增强后: {len(X_train_augmented)}")

    np.random.seed(42)
    sample_size = min(1000, len(X_train_norm))
    original_sample = X_train_norm[np.random.choice(len(X_train_norm), sample_size, replace=False)]
    generated_sample = generate_fake_data(trained_model, sample_size, device=device)

    print("\n[阶段 4/4] 对训练集与测试集分别独立进行滑动窗口打包...")
    train_dataset = create_sliding_window_dataset(
        X_train_augmented, y_train_augmented, seq_len=27, step=5, normal_label=normal_label
    )
    test_dataset = create_sliding_window_dataset(
        X_test_norm, y_test_raw, seq_len=27, step=5, normal_label=normal_label
    )

    return train_dataset, test_dataset, label_encoder, original_sample, generated_sample

def main():
    print("=== 开始执行CAR数据集TabDDPM训练脚本 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 300
    LR = 0.001

    CSV_PATH = "data/car_dataset.csv"
    CACHE_PATH = "data/tabddpm_cache.pth"

    if os.path.exists(CACHE_PATH):
        print(f"\n📦 加载缓存: {CACHE_PATH}")
        checkpoint = torch.load(CACHE_PATH, weights_only=False)
        train_dataset = checkpoint['train_dataset']
        test_dataset = checkpoint['test_dataset']
        label_encoder = checkpoint['label_encoder']
    else:
        print(f"\n🔍 未检测到缓存，开始生成...")
        train_dataset, test_dataset, label_encoder, original_sample, generated_sample = load_augment_and_pack_dataset(CSV_PATH, device, train_ratio=0.8)
        plot_tsne(original_sample, generated_sample, "TabDDPM")
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

    print("\n--- 🚀 开始 RGSNet 训练 ---")
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
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    print("\n--- 评估阶段 ---")
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

    print("\n" + "="*50)
    print("📊 [多分类详细报告]")
    print("="*50)

    report_str = classification_report(
        all_targets,
        all_preds,
        labels=all_indices,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )

    split_idx = report_str.find('\n    accuracy')

    if split_idx != -1:
        print(report_str[:split_idx])
        print("\n")
    else:
        print(report_str)

    bin_targets = (all_targets != 0).astype(int)
    bin_preds = (all_preds != 0).astype(int)
    bin_metrics = precision_recall_fscore_support(bin_targets, bin_preds, labels=[0, 1])

    print("\n" + "="*50)
    print("🛡️ [二分类宏观报告]")
    print("="*50)
    print(f"{'正常类 (0)':<12} | {bin_metrics[0][0]:.4f} | {bin_metrics[1][0]:.4f} | {bin_metrics[2][0]:.4f}")
    print(f"{'攻击类 (1)':<12} | {bin_metrics[0][1]:.4f} | {bin_metrics[1][1]:.4f} | {bin_metrics[2][1]:.4f}")

    torch.save(model.state_dict(), 'TabDDPM/hybrid_cnn_model.pth')
    print("\n模型已保存为 TabDDPM/hybrid_cnn_model.pth")

    print("\n" + "="*70)
    print("🔍 [FGSM 鲁棒性测试]")
    print("="*70)

    epsilon_values = np.linspace(0, 0.1, 11)

    results = evaluate_fgsm_robustness(model, test_dataset, device, epsilon_values)

    plot_metrics_decay_curves(results)

if __name__ == "__main__":
    main()
