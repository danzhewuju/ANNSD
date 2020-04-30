#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/22 13:07
# @Author  : Alex
# @Site    : 
# @File    : clstm_train.py
# @Software: PyCharm


import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from util.util_tool import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

GPU = 0
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 16
# 数据首先需要经过CNN
TIME_STEP = 15  # rnn time step / image height 数据输入的高度
INPUT_SIZE = 32  # rnn input size / image width 数据输入的宽度
LR = 0.001  # learning rate
Resampling = 500  # resampling

# 数据处理
TRAIN_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/train_BDP.csv"
TEST_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/test_BDP.csv"


def collate_fn(data):
    # 主要是用数据的对齐
    data.sort(key=lambda x: x[0].shape[-1], reverse=True)
    max_shape = data[0][0].shape
    labels = []  # 每个数据对应的标签
    length = []  # 记录真实的数目长度
    for i, (d, label) in enumerate(data):
        d_shape = d.shape
        length.append(d.shape[-1])
        if d_shape[-1] < max_shape[-1]:
            tmp_d = np.pad(d, ((0, 0), (0, 0), (0, max_shape[-1] - d_shape[-1])), 'constant')
            data[i] = tmp_d
        else:
            data[i] = d
        labels.append(label)

    return torch.from_numpy(np.array(data)), torch.tensor(labels), length


data_train = Data_info(TRAIN_PATH)
data_test = Data_info(TEST_PATH)
train_data = MyDataset(data_train.data)  # 作为训练集
test_data = MyDataset(data_test.data)  # 作为测试集
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


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
        self.fc1 = nn.Linear(6 * 31 * 32, 32)  # x_ y_ 和你输入的矩阵有关系

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
        bat = x.shape[0]
        if self.gpu is not None:
            res = torch.zeros((bat, 15, 32)).cuda(self.gpu)
        else:
            res = torch.zeros((bat, 15, 32))
        for i in range(bat):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] / Resampling
            for j in range(int(length)-1):
                tmp_split = tmp_x[:, Resampling*j:(j+1)*Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, Resampling))
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


clstm = clstm(gpu=0).cuda(0)
optimizer = torch.optim.Adam(clstm.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

acc = []
for epoch in range(EPOCH):
    for step, (b_x, b_y, length) in enumerate(train_loader):  # gives batch data
        b_x_g = b_x.cuda(0)
        b_y_g = b_y.cuda(0)
        # b_x = b_x.view(-1, 100, 1000)  # reshape x to (batch, time_step, input_size)
        output = clstm(b_x_g)  # rnn output
        loss = loss_func(output, b_y_g)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        pred_y = torch.max(output, 1)[1].data
        pred_y = pred_y.cpu()
        res_tmp = [1 if pred_y[i] == b_y[i] else 0 for i in range(len(b_y))]
        acc += res_tmp
        if step > 0 and step % 50 == 0:
            accuracy = sum(acc) / len(acc)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.6f' % accuracy)
            acc.clear()
            torch.save(clstm.state_dict(), "../save_model/auto_encoder_lstm.pkl")
            print("step:{} 模型保存成功！".format(step))

    print("模型被正常保存！")
