


# class CNN_Discriminator(nn.Module):
#     def __init__(self, input_size, num_channels, kernel_size, stride, padding):
#         super(CNN_Discriminator, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.conv3 = nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels*4, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.pool = nn.MaxPool1d(2)
#         self.fc1 = nn.Linear(num_channels*4, 1)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = x.view(-1, self.num_flat_features(x))
#         x = torch.sigmoid(self.fc1(x))
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

# import torch as nn
from torch import nn
class Dis_CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dis_CNN, self).__init__()

        # self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(hidden_size * (input_size // 4), hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 1)
        # self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.LeakyReLU(0.2)
        # self.deconv1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=1)
        # self.d_bn1 = nn.BatchNorm2d(out_channels * 4)
        # self.deconv2 = nn.ConvTranspose1d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=1)
        # self.d_bn2 = nn.BatchNorm2d(out_channels * 2)
        # self.deconv3 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=1)
        # self.d_bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        # x = self.d_bn1(self.relu(self.deconv1(x)))
        # x = self.d_bn2(self.relu(self.deconv2(x)))
        # x = self.d_bn3(self.relu(self.deconv3(x)))
        # print("x",x)
        # x = x.view(x.size(0), -1)
        return x
