import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_SigFucker_layer3(nn.Module):
    def __init__(self, num_classes):
        super(CNN_SigFucker_layer3, self).__init__()
        
        # 第一层卷积：2个输入通道，16个输出通道，卷积核大小为7，步长1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        # 第二层卷积：16个输入通道，32个输出通道，卷积核大小为5，步长1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # 第三层卷积：32个输入通道，64个输出通道，卷积核大小为3，步长1
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # 输入数据维度为 [batch_size, 2, 2048]
        
        # 第一层卷积和池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 输出维度：[batch_size, 16, 1024]
        
        # 第二层卷积和池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 输出维度：[batch_size, 32, 512]
        
        # 第三层卷积和池化
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 输出维度：[batch_size, 64, 256]
        
        # 展平
        x = x.view(x.size(0), -1)  # 展平维度：[batch_size, 64 * 256]
        
        # 全连接层
        x = F.relu(self.fc1(x))    # 输出维度：[batch_size, 128]
        x = self.fc2(x)             # 输出维度：[batch_size, num_classes]
        
        return x

# ciallo: Run this Fucking .py to verify the construction
if __name__ == '__main__':
    data = torch.randn(128, 2, 2048)    # ciallo: [batch_size, channel(defalut is 1), I and Q channel(So is 2), Sample_Length]
    model = CNN_SigFucker_layer3(num_classes=5)
    print("模型结构如下喵: {}".format(model), end='\n')                           # ciallo: Print model constuction and Args
    out = model(data)
    print("模型输出维度如下喵: {}".format(out.shape), end='\n')
