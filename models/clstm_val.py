#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/22 17:08
# @Author  : Alex
# @Site    : 
# @File    : clstm_val.py
# @Software: PyCharm

from  .clstm_train import clstm
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
Resampling  = 500  # resampling


VAL_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/train.csv"
model_path = "./save_model/clstm.pkl"
data_val = Data_info(VAL_PATH)
val_data = MyDataset(data_val.data)  # 作为验证集
train_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

clstm = clstm()
clstm.load_state_dict(torch.load(model_path))
