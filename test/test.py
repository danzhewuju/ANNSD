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

if __name__ == '__main__':
    # res = generate_data(duration=1200, wins=(2, 16), k=3000)
    # print(res)
    # print(Counter(res))
    # a = np.random.randint(0, 10)
    # print(a)
    s_1= uuid.uuid1()
    print(s_1)
    channel_path="/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/data_slice/channels_info/BDP_seq.csv"
    data = pd.read_csv(channel_path, sep=',')
    names = list(data['chan_name'])
    print(names)

