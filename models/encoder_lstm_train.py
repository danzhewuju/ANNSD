#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/22 13:07
# @Author  : Alex
# @Site    :
# @File    : encoder_lstm_train.py
# @Software: PyCharm


import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from util.util_tool import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

GPU = 0


def collate_fn(data):  #
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


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(100 * 500, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.Tanh(),
            nn.Linear(512, 100 * 500),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


# 数据处理
data_train = Data_info(TRAIN_PATH)
data_test = Data_info(TEST_PATH)
train_data = MyDataset(data_train.data)  # 作为训练集
test_data = MyDataset(data_test.data)  # 作为测试集
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

encoder_path = "../save_model/autoencoder_back.pkl"
encoder = AutoEncoder().cuda(GPU)
encoder.load_state_dict(torch.load(encoder_path))

rnn = RNN().cuda(GPU)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

acc = []
for epoch in range(EPOCH):
    for step, (b_x, b_y, length) in enumerate(train_loader):  # gives batch data
        res_x = torch.zeros((BATCH_SIZE, TIME_STEP, INPUT_SIZE))
        res_x = res_x.cuda(GPU)
        for i in range(b_x.shape[0]):  # batch size
            x = b_x[i][0]
            # x.reshape(x.shape[1], x.shape[2])
            for j in range(0, length[i] - 500, 500):
                tmp_x = x[:, j:j + 500]
                tmp_x = torch.reshape(tmp_x, (1, 100 * 500))
                tmp_x = tmp_x.cuda(GPU)
                code, decode = encoder(tmp_x)
                res_x[i][j // 500] = code

        # b_x_g = res_x.cuda(0)
        b_y_g = b_y.cuda(0)
        # b_x = b_x.view(-1, 100, 1000)  # reshape x to (batch, time_step, input_size)
        output = rnn(res_x)  # rnn output
        loss = loss_func(output, b_y_g)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        pred_y = torch.max(output, 1)[1].data
        pred_y = pred_y.cpu()
        res_tmp = [1 if pred_y[i] == b_y[i] else 0 for  i in range(len(pred_y))]
        acc += res_tmp
        if step % 100 == 0:
            accuracy = sum(acc) / len(acc)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            acc.clear()
            torch.save(rnn.state_dict(), "../save_model/auto_encoder_lstm.pkl")
            print("step:{} 模型保存成功！".format(step))


    print("模型被正常保存！")
