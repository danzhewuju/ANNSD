import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
sys.path.append('../')
from util.util_tool import Data_info, MyDataset

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 16
LR = 0.005  # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5
GPU = 0

TRAIN_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/train_BDP.csv"
TEST_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/test_BDP.csv"


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


data_train = Data_info(TRAIN_PATH)
data_test = Data_info(TEST_PATH)
train_data = MyDataset(data_train.data)  # 作为训练集
test_data = MyDataset(data_test.data)  # 作为测试集
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


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


autoencoder = AutoEncoder().cuda(GPU)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, b_label, length) in enumerate(train_loader):
        # x, b_label = x.cuda(GPU), b_label.cuda(GPU)
        # 需要对X 的输入进行处理
        n = sum(length) // 500
        x_split = torch.zeros((n, 100, 500))
        count = 0
        for index, x_x in enumerate(x):
            for i in range(0, length[index]-500, 500):
                tmp_x = x_x[:, :, i:i + 500]
                tmp_x.reshape(-1, 100 * 500)
                x_split[count] = tmp_x
                count += 1

        # x_input = torch.from_numpy(x_split)

        b_x = x_split.view(-1, 100 * 500)  # batch x, shape (batch, 28*28)
        b_y = b_x.clone()  # batch y, shape (batch, 28*28)
        b_x = b_x.cuda(GPU)
        b_y = b_y.cuda(GPU)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)  # mean square error
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            if GPU >= 0:
                # 使用GPU
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
            else:
                # 不使用GPU
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            torch.save(autoencoder.state_dict(), '../save_model/autoencoder.pkl')
            print("step:{} 模型保存成功！".format(step))