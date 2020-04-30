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
from util.util_tool import Data_info, MyDataset,collate_fn



if __name__ == '__main__':
    BATCH_SIZE = 16
    TEST_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/test_BDP.csv"
    VAL_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/val_BDP.csv"
    data_val = Data_info(VAL_PATH)
    val_data = MyDataset(data_val.data)  # 作为测试集
    # data = val_data[1]
    # print(data)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    data_iter = list(val_loader)
    d = data_iter[1]
    print(d)
