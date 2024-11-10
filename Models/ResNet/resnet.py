import torch
import torch.nn as nn
import torch.nn.functional as functional

import torchvision.models as vision_models

# ciallo GPT: 这里的planes是通道捏
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block: list[int], num_classes: int):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(2, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_block[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))

        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, batch_x):
        batch_x = self.conv1(batch_x)
        batch_x = self.bn1(batch_x)
        batch_x = self.relu(batch_x)
        batch_x = self.max_pool(batch_x)
        batch_x = self.layer1(batch_x)
        batch_x = self.layer2(batch_x)
        batch_x = self.layer3(batch_x)
        batch_x = self.layer4(batch_x)
        batch_x = self.avgpool(batch_x)
        batch_x = batch_x.flatten(1)
        batch_x = self.fc(batch_x)
        return batch_x


def resnet18(num_classes):
    network = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return network

def vision_resnet18(num_classes):
    network = vision_models.resnet18()
    network.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, num_classes)
    return network