#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/31 20:20
# @Author  : Alex
# @Site    : 
# @File    : test.py
# @Software: PyCharm
# from data.generatedata import generate_data
import numpy as np
from collections import Counter
import pandas as pd
import uuid
import re
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from util.util_tool import Data_info, MyDataset, collate_fn

if __name__ == '__main__':
    TRAIN_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/train_BDP.csv"
    datainfo = Data_info(TRAIN_PATH)
    p = datainfo.next_batch_data(16)
    print(p)
    mydataset = MyDataset(next(p))
    dataLoader = DataLoader(mydataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
    for ima, l , _ in dataLoader:
        print(ima, l)
    # a = np.random.randint(0, 5, 1)
    # print(a)
    d = datainfo.next_batch_data(16)
    data = MyDataset(next(d))
    dataLoader = DataLoader(data, shuffle=True, batch_size=16, collate_fn = collate_fn)
    ima, l, _ =  dataLoader
    print("IMA, l")

    # a = np.random.randint(0, 5, 1)
    # # print(a)
    # a = 0
    # if a :
    #     print("True")
    # else:
    #     print("False")
    # a = torch.randn((5, 2))
    # b = torch.max(a, 1).values
    # print(b)

