import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ciallo GPT: 一个RFF层来生成随机傅里叶特征
class RFFLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RFFLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = nn.Parameter(torch.rand(output_dim) * 2 * np.pi)

    def forward(self, x):
        return torch.cos(x @ self.W + self.b)

# ciallo GPT: 网络构建
class DualRFFCNN2(nn.Module):
    def __init__(self, input_dim):
        super(DualRFFCNN2, self).__init__()
        
        # RFF层1
        self.rff1 = RFFLayer(input_dim, 256)
        
        # 卷积层1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RFF层2
        self.rff2 = RFFLayer(32 * 128, 256)  # 适应性调整

        # 卷积层2
        self.conv2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # 适应性调整
        self.fc2 = nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        # RFF层1
        x = self.rff1(x)
        x = x.view(-1, 1, 16, 256)  # 调整形状

        # 卷积层1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # RFF层2
        x = self.rff2(x.view(x.size(0), -1))  # 扁平化处理
        x = x.view(-1, 1, 16, 256)  # 调整形状

        # 卷积层2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 扁平化
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ciallo GPT: 在之前网络的基础上添加了部分特征处理喵
#             对提取出的特征进行纵向拼接,融合,并进行横向均值池化降维
#             最后也添加了全连接和softmax层用来五分类喵
class DualRFFCNN2WithClassification(nn.Module):
    def __init__(self, input_dim):
        super(DualRFFCNN2WithClassification, self).__init__()
        
        # RFF层1
        self.rff1 = RFFLayer(input_dim, 256)
        
        # 卷积层1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RFF层2
        self.rff2 = RFFLayer(32 * 128, 256)  # 适应性调整

        # 卷积层2
        self.conv2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # 适应性调整
        self.fc2 = nn.Linear(128, 10)  # 输出层
        
        # 特征融合层
        self.fc_fusion = nn.Linear(128, 64)  # 可以根据需求调整输出维度
        
        # 分类层
        self.fc_classify = nn.Linear(64, 5)  # 输出5个类别

    def forward(self, x):
        # RFF层1
        x1 = self.rff1(x)
        x1 = x1.view(-1, 1, 16, 256)  # 调整形状

        # 卷积层1
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool1(x1)

        # RFF层2
        x2 = self.rff2(x1.view(x1.size(0), -1))  # 扁平化处理
        x2 = x2.view(-1, 1, 16, 256)  # 调整形状

        # 卷积层2
        x2 = F.relu(self.conv2(x2))
        x2 = self.pool2(x2)

        # 扁平化
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)

        # 纵向拼接
        concatenated_features = torch.cat((x1_flat, x2_flat), dim=1)  # 按列拼接

        # 特征融合
        fused_features = F.relu(self.fc_fusion(concatenated_features))

        # 横向均值池化降维
        # 假设 fused_features 的形状为 (batch_size, feature_dim)
        pooled_features = torch.mean(fused_features, dim=1, keepdim=True)  # 计算均值
        
        # 输入全连接层
        x = F.relu(self.fc_fusion(pooled_features))

        # 分类层，输出五分类的结果
        x = self.fc_classify(x)
        
        return x  # 返回 logits
