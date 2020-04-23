#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/22 17:08
# @Author  : Alex
# @Site    : 
# @File    : clstm_val.py
# @Software: PyCharm

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from util.util_tool import *
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
# 数据首先需要经过CNN
TIME_STEP = 15  # rnn time step / image height 数据输入的高度
INPUT_SIZE = 100  # rnn input size / image width 数据输入的宽度
LR = 0.001  # learning rate
Resampling = 500  # resampling


class clstm(nn.Module):
    def __init__(self, gpu=None):
        super(clstm, self).__init__()
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
        self.fc1 = nn.Linear(6 * 31 * 32, 100)  # x_ y_ 和你输入的矩阵有关系

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
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
        if self.gpu is not None:
            res = torch.zeros((1, 15, 100)).cuda(self.gpu)
        else:
            res = torch.zeros((1, 15, 100))
        length = x.size(-1) / Resampling
        for i in range(int(length)):
            tmx = x[:, :, :, i * 500:(i + 1) * 500]
            tmx = self.layer1(tmx)
            tmx = self.layer2(tmx)
            tmx = self.layer3(tmx)
            tmx = self.layer4(tmx)
            tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
            tmx = self.fc1(tmx)
            res[0][i] = tmx

        r_out, (h_n, h_c) = self.rnn(res, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


VAL_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/val.csv"
model_path = "./save_model/clstm.pkl"
data_val = Data_info(VAL_PATH)
val_data = MyDataset(data_val.data)  # 作为验证集
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

clstm = clstm(gpu=0).cuda(0)
clstm.load_state_dict(torch.load(model_path))
acc = []
count = 1000
for step, (b_x, b_y) in enumerate(val_loader):
    if step < count:
        b_x, b_y = b_x.cuda(0), b_y.cuda(0)
        test_output = clstm(b_x)  # (samples, time_step, input_size)
        pred_y = torch.max(test_output, 1)[1].data
        if pred_y[0] == b_y[0]:
            acc.append(1)
        else:
            acc.append(0)
    else:
        break
print(np.mean(acc))
