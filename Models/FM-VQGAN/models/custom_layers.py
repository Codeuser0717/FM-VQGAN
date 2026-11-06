import torch
import torch.nn as nn

class CompConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(CompConv3x3, self).__init__()
        self.split_channels = in_channels // 2
        self.needs_proj = (in_channels != out_channels) or (stride != 1)
        if self.needs_proj:
            self.proj = nn.Conv2d(self.split_channels, out_channels - out_channels//2, kernel_size=1, stride=stride)
            self.conv = nn.Conv2d(in_channels - self.split_channels, out_channels//2, kernel_size=3, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.split_channels, x.size(1) - self.split_channels], dim=1)
        out1 = self.proj(x1) if self.needs_proj else x1
        out2 = self.conv(x2)
        return torch.cat([out1, out2], dim=1)

class GRN(nn.Module):
    def __init__(self, dim):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(1, -1, 1, 1) + x
