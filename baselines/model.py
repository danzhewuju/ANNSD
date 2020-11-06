import torch
from torch import nn


# import sys
#
# sys.path.append('../')
# from util.util_tool import


class clstm(nn.Module):
    def __init__(self, gpu=None, input_size=32, Resampling=500):
        super(clstm, self).__init__()
        self.input_size = input_size
        self.Resampling = Resampling
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(6 * 31 * 32, 32)  # x_ y_ 和你输入的矩阵有关系

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=self.input_size,  # 输入向量的长度
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 2)
        self.gpu = gpu

    def forward(self, x):
        # res = []
        # batch_size = x.size(0)
        # 需要对数据进行处理
        bat = x.shape[0]
        if self.gpu is not None:
            res = torch.zeros((bat, 15, 32)).cuda(self.gpu)
        else:
            res = torch.zeros((bat, 15, 32))
        for i in range(bat):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.fc1(tmx)
                res[i][j] = tmx

        r_out, (h_n, h_c) = self.rnn(res, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


class cnnVoting(nn.Module):

    def __init__(self, gpu=0, input_size=32, Resampling=500):
        super(cnnVoting, self).__init__()
        self.input_size = input_size
        self.Resampling = Resampling
        self.gpu = gpu

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(6 * 31 * 32, 32),  # x_ y_ 和你输入的矩阵有关系
            nn.Linear(32, 2)
        )

    '''
    需要对于数据进行处理;
    X=(batch_size, weight, height)
    '''

    def forward(self, x):
        batch_size = x.shape[0]
        if self.gpu >= 0:
            res = torch.zeros(batch_size, 2).cuda(self.gpu)
        else:
            res = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.layer5(tmx)
                tmx = tmx.reshape(-1)
                res[i] += tmx
            res[i] /= length
        return res
