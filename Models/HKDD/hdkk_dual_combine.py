import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN模块用于从IQ数据中提取特征
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1)  # 2通道IQ数据
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.squeeze(-1)  # 返回的特征形状为 (batch_size, 256)

# DNN模块用于处理手工设计的特征
class DNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(DNNFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x  # 返回的特征形状为 (batch_size, 32)

# 注意力机制模块
class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * attn_weights  # 返回经过加权后的特征

# 整体HKDD网络
class HKDDNet(nn.Module):
    def __init__(self, handcrafted_feature_dim):
        super(HKDDNet, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor()
        self.dnn_extractor = DNNFeatureExtractor(handcrafted_feature_dim)
        self.attention = AttentionModule(input_dim=256 + 32)  # 联合特征维度
        self.classifier = nn.Linear(256 + 32, 10)  # 假设有10种调制类型

    def forward(self, iq_data, handcrafted_features):
        cnn_features = self.cnn_extractor(iq_data)
        dnn_features = self.dnn_extractor(handcrafted_features)
        fc = torch.cat((cnn_features, dnn_features), dim=1)  # 垂直拼接
        fa = self.attention(fc)  # 通过注意力机制
        out = self.classifier(fa)  # 分类层
        return F.log_softmax(out, dim=1)  # 输出每类的概率

if __name__ == '__main__':
    # 测试网络
    handcrafted_feature_dim = 100  # 手工设计特征的维度
    model = HKDDNet(handcrafted_feature_dim)

    # 假设IQ数据为 (batch_size, 2, 2048) 的张量，手工特征为 (batch_size, 100)
    iq_data = torch.randn(32, 2, 2048)
    handcrafted_features = torch.randn(32, handcrafted_feature_dim)

    # 前向传播测试
    output = model(iq_data, handcrafted_features)
    print(output.shape)  # 输出应为 (batch_size, 10)，即每个样本的分类概率
