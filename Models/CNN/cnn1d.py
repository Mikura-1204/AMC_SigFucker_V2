#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - This Fucking Net is provided by Signal-Fucking-Organazition.
#   Used to classify 5 classes Fucking Radio-Sig
#
# - GPT: 这个网络的主要目的是对一维信号进行分类，通常用于处理时序数据，比如音频或其他时间序列信号。
#        它包含多个卷积层和残差连接，能够有效提取特征并缓解深度网络训练中的梯度消失问题
#####------------------------------------------------------------------------------------#####

import torch.nn as nn
import torch
import torch.nn.functional as F

# n_class = 5

# ResNet {{{
class ResNet1D(nn.Module):
    def __init__(self,n_class):
        super(ResNet1D, self).__init__()
        
        # ciallo: Change the Fucking input Conv to one-dim
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(2,5), padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        
        # ciallo: try to change the best kernel_size = 5 FFFFUCK IT MAN !
        # self.conv1 = nn.Conv1d(2, 64, kernel_size=5, padding=0, bias=False)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        # self.pool = nn.MaxPool2d(kernel_size=(1,5), stride=(1,2))
        self.conv2 = ResidualStack(64, kernel_size=3, pool_size=2)
        self.conv3 = ResidualStack(64, kernel_size=3, pool_size=2)
        
        # ciallo: change the Fucking Resconnect
        # self.conv4 = ResidualStack(64, kernel_size=3, pool_size=2)
        # self.conv5 = ResidualStack(64, kernel_size=3, pool_size=2)

        # ciallo: Change the Fucking FC layer dim, using Auto-Calculate
        # self.fc = nn.Linear(9984, n_class)
        self._initialize_fc_layer(n_class)
    def _initialize_fc_layer(self, n_class):
        # 通过输入虚拟数据来确定 fc 层的输入尺寸
        with torch.no_grad():
            x = torch.randn(1, 2, 2048)  # 输入的虚拟数据
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.conv3(x)
            # x = self.conv4(x)
            # x = self.conv5(x)
            flattened_size = x.view(1, -1).size(1)
        
        # 根据计算结果设置 fc 层的输入尺寸
        self.fc = nn.Linear(flattened_size, n_class)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # ciallo: Change the Fucking squeeze
        x = x.squeeze(2)
        # print(x.shape,2)
        x = self.conv2(x)
        # print(x.shape,3)
        x = self.conv3(x).view(x.size(0), -1)
        # print(x.shape,4)
        # x = self.conv4(x)
        # print(x.shape,5)
        # x = self.conv5(x).view(x.size(0), -1)
        # print(x.shape,6)
        x = self.fc(x)
        # print(x.shape,7)
        return x

# ciallo GPT: 使用多个 ResidualStack 来构建残差块。每个 ResidualStack 中包含多个卷积层和跳跃连接（shortcut connection），
#             通过添加输入和输出的特征来缓解深度网络的训练难度。每个残差模块后都有一个池化层，降低特征图的尺寸，保留重要信息。
class ResidualStack(nn.Module):
    def __init__(self, in_channel, kernel_size, pool_size, first=False):
        super(ResidualStack, self).__init__()
        mid_channel = 64
        padding = 1
        conv = nn.Conv1d
        pool = nn.MaxPool1d
        self.conv1 = conv(in_channel, mid_channel, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channel)
        self.conv2 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channel)
        self.conv3 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm1d(mid_channel)
        self.conv4 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn4 = nn.BatchNorm1d(mid_channel)
        self.conv5 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm1d(mid_channel)
        self.pool = pool(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        # residual 1
        x = self.conv1(x)
        x = self.bn1(x)
        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # x += shortcut
        x = F.relu(x)

        # residual 2
        shortcut = x
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        # x += shortcut
        x = F.relu(x)
        x = self.pool(x)
        # print(x.shape,2)
        return x


def resnet1d(**kwargs):
    return ResNet1D(**kwargs)

# ciallo: If you run this Fucking single code as below, you can simply see the construction of this Fucking Net
#         Also the input should be (batch_size, 1, 2, 2500) 
#         But now I have modified this Fucking-Net to adapt my Shit-Type5 dataset 
if __name__ == '__main__':
    data = torch.randn(128, 2, 2048)    # ciallo: [batch_size, channel(defalut is 1), I and Q channel(So is 2), Sample_Length]
    model = resnet1d(n_class=5)
    print("模型结构如下喵: {}".format(model), end='\n')                           # ciallo: Print model constuction and Args
    out = model(data)
    print("模型输出维度如下喵: {}".format(out.shape), end='\n')
