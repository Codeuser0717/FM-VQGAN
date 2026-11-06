import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .custom_layers import CompConv3x3, GRN

class RGStageBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(RGStageBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CompConv3x3(planes, planes, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.grn = GRN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.grn(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out

class RGSNet(nn.Module):
    def __init__(self, num_classes=15):
        super(RGSNet, self).__init__()

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.branch1 = nn.Sequential(*list(resnet18.children())[:-1])

        self.branch2_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch2_bn1 = nn.BatchNorm2d(64)
        self.branch2_layer1 = RGStageBlock(64, 64)
        self.branch2_layer2 = RGStageBlock(256, 128, stride=2)
        self.branch2_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 + 512, num_classes)

    def forward(self, x):
        x_large = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        feat1 = self.branch1(x_large)
        feat1 = feat1.view(feat1.size(0), -1)

        feat2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        feat2 = self.branch2_layer1(feat2)
        feat2 = self.branch2_layer2(feat2)
        feat2 = self.branch2_avgpool(feat2)
        feat2 = feat2.view(feat2.size(0), -1)

        fused_feat = torch.cat([feat1, feat2], dim=1)
        return self.fc(fused_feat)
