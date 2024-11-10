import torch
import torch.nn as nn

'''《A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition》'''

class MCLDNN(nn.Module):
    def __init__(self, num_classes):
        super(MCLDNN, self).__init__()
        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(2, 8), padding="same"),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )
        self.conv2_3_pad = nn.ZeroPad1d((7, 0))
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8),
            nn.BatchNorm1d(50),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8),
            nn.BatchNorm1d(50),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 8), padding="same"),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(2, 5)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        # Part-B: TRemporal Characteristics Extraction Section
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        # Part-C: Fully Connected Classifier
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(),
        )
        self.softmax = nn.Sequential(
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, batch_x):
        conv2_in, conv3_in = batch_x[:, 0:1], batch_x[:, 1:2]
        conv2_in, conv3_in = self.conv2_3_pad(conv2_in), self.conv2_3_pad(conv3_in)
        conv2_out, conv3_out = self.conv2(conv2_in), self.conv3(conv3_in)
        conv2_out, conv3_out = torch.unsqueeze(conv2_out, dim=2), torch.unsqueeze(conv3_out, dim=2)
        
        concatenate1 = torch.concatenate((conv2_out, conv3_out), dim=2)
        conv4_out = self.conv4(concatenate1)
        
        batch_x = torch.unsqueeze(batch_x, dim=1)
        conv1_out = self.conv1(batch_x)
        concatenate2 = torch.concatenate((conv4_out, conv1_out), dim=1)
        conv5_out = self.conv5(concatenate2)
        conv5_out = conv5_out.permute(0, 3, 2, 1)
        conv5_out = conv5_out.flatten(2)
        
        outputs, _ = self.lstm1(conv5_out)
        outputs, _ = self.lstm2(outputs)
        
        outputs = self.fc1(outputs[:, -1])
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        
        return outputs