import torch
from torch import nn
import torch.nn.functional as F

'''
《Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors》
'''

# ciallo GPT: 芝士简单的双层 LSTM 网络，最后接一个全连接层和一个 Softmax 层
class LSTM(nn.Module):
    def __init__(self, classes):
        super(LSTM, self).__init__()
        
        # ciallo: PAY ATTEIONTION that the first Dim of input is Batch-Size
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=128)
        self.softmax = nn.Linear(in_features=128, out_features=classes)
        
    def forward(self, batch_x):
        
        # ciallo GPT: 这一步将输入数据的维度顺序从 (batch_size, seq_len, features) 转换为 (batch_size, features, seq_len)
        batch_x = batch_x.permute(0, 2, 1)
        
        # ciallo GPT: 第二个返回值 _ 是隐藏状态和细胞状态，这里不需要，所以用 _ 忽略
        batch_x, _ = self.lstm1(batch_x)
        batch_x, _ = self.lstm2(batch_x)
        batch_x = self.fc(batch_x[:, -1])
        batch_y = self.softmax(batch_x)

        return batch_y
