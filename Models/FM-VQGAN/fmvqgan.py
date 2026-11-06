import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(labels, num_classes, device):
    return F.one_hot(labels.long(), num_classes=num_classes).float().to(device)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=16, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            - 2 * z_e @ self.embedding.weight.t()
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)
        codebook_loss   = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=input_dim + num_classes,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.fc_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, y_onehot):
        x_cond = torch.cat([x, y_onehot], dim=1)
        x_seq  = x_cond.unsqueeze(1)
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(x_seq)
        f0 = self.lstm_norm(lstm_out.squeeze(1))
        f1 = self.fc_block1(f0)
        f2 = self.fc_block2(f1)
        validity = self.output_layer(f2)
        return validity, [f0, f1, f2]


class ConditionalFMVQGAN_Generator(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64,
                 num_embeddings=128, embedding_dim=16):
        super().__init__()
        self.num_classes   = num_classes
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, y_onehot):
        z_e = self.encoder(x)
        z_q, vq_loss = self.vq(z_e)
        z_q_cond = torch.cat([z_q, y_onehot], dim=1)
        x_recon = self.decoder(z_q_cond)
        return x_recon, vq_loss

    def generate_fake_from_noise(self, y_onehot, device):
        batch_size = y_onehot.size(0)
        noise = torch.randn(batch_size, self.embedding_dim, device=device)
        noise_cond = torch.cat([noise, y_onehot], dim=1)
        return self.decoder(noise_cond)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def feature_matching_loss(real_features: list, fake_features: list) -> torch.Tensor:
    fm_loss = torch.tensor(0.0, device=real_features[0].device)
    for f_real, f_fake in zip(real_features, fake_features):
        f_real_norm = F.normalize(f_real, p=2, dim=1)
        f_fake_norm = F.normalize(f_fake, p=2, dim=1)
        fm_loss = fm_loss + F.mse_loss(f_real_norm.mean(0), f_fake_norm.mean(0))
    return fm_loss


def compute_gradient_penalty_cond(D, real_samples, fake_samples, y_onehot, device):
    alpha = torch.rand(real_samples.size(0), 1, device=device).expand(real_samples.size())
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = D(interpolates, y_onehot)
    fake = torch.ones(real_samples.shape[0], 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True,
        retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def sample_balanced_labels(num_classes, batch_size, device, minority_classes=None):
    if minority_classes is None:
        candidate = torch.arange(num_classes, device=device)
    else:
        candidate = torch.tensor(minority_classes, device=device)
    idx = torch.randint(0, len(candidate), (batch_size,), device=device)
    return candidate[idx]


def train_conditional_fmvqgan(
    full_data_tensor,
    full_labels_tensor,
    num_classes,
    minority_classes,
    epochs=300,
    batch_size=256,
    device='cuda',
    lambda_gp=10,
    lambda_adv=0.1,
    lambda_fm=1.0,
    lambda_cls=0.5,
    n_critic=3,
    drift_eps=0.01,
):
    input_dim = full_data_tensor.shape[1]

    G = ConditionalFMVQGAN_Generator(input_dim=input_dim, num_classes=num_classes).to(device)
    D = ConditionalDiscriminator(input_dim=input_dim, num_classes=num_classes).to(device)
    C = AuxiliaryClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=2e-5, betas=(0.5, 0.9), weight_decay=1e-4)
    scheduler_C = torch.optim.lr_scheduler.StepLR(opt_C, step_size=30, gamma=0.5)

    class_counts = torch.bincount(full_labels_tensor.long(), minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[full_labels_tensor.long()]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(full_data_tensor), replacement=True
    )
    dataset    = torch.utils.data.TensorDataset(full_data_tensor, full_labels_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=True
    )

    ce_loss = nn.CrossEntropyLoss()
    G.train(); D.train(); C.train()

    for epoch in range(epochs):
        for real_batch, real_labels in dataloader:
            current_batch_size = real_batch.size(0)
            real_batch  = real_batch.to(device)
            real_labels = real_labels.to(device).long()
            real_y_oh   = to_one_hot(real_labels, num_classes, device)

            for _ in range(n_critic):
                opt_D.zero_grad()
                real_validity, _ = D(real_batch, real_y_oh)
                fake_labels = sample_balanced_labels(
                    num_classes, current_batch_size, device, minority_classes
                )
                fake_y_oh  = to_one_hot(fake_labels, num_classes, device)
                fake_batch = G.generate_fake_from_noise(fake_y_oh, device).detach()
                fake_validity, _ = D(fake_batch, fake_y_oh)

                gp = compute_gradient_penalty_cond(
                    D, real_batch.data, fake_batch.data, real_y_oh, device
                )

                drift_penalty = torch.mean(real_validity ** 2)

                d_loss = (
                    -torch.mean(real_validity) + torch.mean(fake_validity)
                    + lambda_gp * gp
                    + drift_eps * drift_penalty
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_D.step()

            opt_C.zero_grad()
            C.train()
            logits_real = C(real_batch)
            cls_loss = ce_loss(logits_real, real_labels)
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(C.parameters(), max_norm=1.0)
            opt_C.step()

            opt_G.zero_grad()

            recon_batch, vq_loss = G(real_batch, real_y_oh)
            recon_loss = F.mse_loss(recon_batch, real_batch)

            fake_labels_g = sample_balanced_labels(
                num_classes, current_batch_size, device, minority_classes
            )
            fake_y_oh_g      = to_one_hot(fake_labels_g, num_classes, device)
            fake_batch_for_g = G.generate_fake_from_noise(fake_y_oh_g, device)

            fake_validity_for_g, fake_features = D(fake_batch_for_g, fake_y_oh_g)
            with torch.no_grad():
                _, real_features = D(real_batch, real_y_oh)

            g_adv_loss = -torch.mean(fake_validity_for_g)
            fm_loss    = feature_matching_loss(real_features, fake_features)

            C.eval()
            for p in C.parameters(): p.requires_grad = False
            logits_fake = C(fake_batch_for_g)
            cls_g_loss  = ce_loss(logits_fake, fake_labels_g)
            for p in C.parameters(): p.requires_grad = True
            C.train()

            g_loss = (
                recon_loss
                + vq_loss
                + lambda_adv * g_adv_loss
                + lambda_fm  * fm_loss
                + lambda_cls * cls_g_loss
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

        scheduler_C.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                d_real_mean = real_validity.mean().item()
                d_fake_mean = fake_validity.mean().item()
            print(
                f"[Epoch {epoch:>4d}/{epochs}] "
                f"D: {d_loss.item():>8.4f} | "
                f"G: {g_loss.item():>7.4f} | "
                f"D(r): {d_real_mean:>7.3f} | "
                f"D(f): {d_fake_mean:>7.3f} | "
                f"Rec: {recon_loss.item():>5.3f} | "
                f"VQ: {vq_loss.item():>5.3f} | "
                f"FM: {fm_loss.item():>6.4f} | "
                f"Cls(C): {cls_loss.item():>5.3f} | "
                f"Cls(G): {cls_g_loss.item():>5.3f}"
            )

    print("\nConditional FMVQGAN 训练完成！")
    torch.save(G.state_dict(), 'cond_fmvqgan_generator.pth')
    torch.save(D.state_dict(), 'cond_fmvqgan_discriminator.pth')
    torch.save(C.state_dict(), 'cond_fmvqgan_classifier.pth')
    return G, D, C


def generate_samples_by_class(G, target_class, num_samples, num_classes, device):
    G.eval()
    with torch.no_grad():
        target_labels = torch.full((num_samples,), target_class, dtype=torch.long, device=device)
        y_oh = to_one_hot(target_labels, num_classes, device)
        fake_samples = G.generate_fake_from_noise(y_oh, device)
    return fake_samples.cpu().numpy()
