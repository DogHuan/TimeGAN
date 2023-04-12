import torch as nn

#卷积模块，由卷积核和激活函数组成
class conv_block(nn.Module):
    def __init__(self, ks, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=ks,stride=1,padding=1,bias=True),  #二维卷积核，用于提取局部的图像信息
            nn.ReLU(inplace=True), #这里用ReLU作为激活函数
            nn.Conv2d(ch_out, ch_out, kernel_size=ks,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)


# 常规CNN模块（由几个卷积模块堆叠而成）
class CNN(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(CNN, self).__init__()
        feature_list = [16, 32, 64, 128, 256]  # 代表每一层网络的特征数，扩大特征空间有助于挖掘更多的局部信息
        self.conv1 = conv_block(kernel_size, in_ch, feature_list[0])
        self.conv2 = conv_block(kernel_size, feature_list[0], feature_list[1])
        self.conv3 = conv_block(kernel_size, feature_list[1], feature_list[2])
        self.conv4 = conv_block(kernel_size, feature_list[2], feature_list[3])
        self.conv5 = conv_block(kernel_size, feature_list[3], feature_list[4])
        self.fc = nn.Sequential(  # 全连接层主要用来进行分类，整合采集的局部信息以及全局信息
            nn.Linear(feature_list[4] * 28 * 28, 1024),  # 此处28为MINST一张图片的维度
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        device = x.device
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.view(x5.size()[0], -1)  # 全连接层相当于做了矩阵乘法，所以这里需要将维度降维来实现矩阵的运算
        out = self.fc(x5)
        return out

# import torch
# import torch.nn as nn
#
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
