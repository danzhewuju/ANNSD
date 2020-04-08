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
import random

if __name__ == '__main__':
    a = np.random.randint(0, 100, (100, 999))
    b = np.pad(a, ((0,0), (0, 1)), constant_values=(0, 0))
    print(b.shape)
    a = [1, 2, 3 ]
    print(np.mean(a))
    # print(a)
    # print(b)
