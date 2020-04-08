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
    a = list(range(100))
    b = list(range(100))

    time_seed = random.randint(0, 100)
    random.seed(time_seed)
    random.shuffle(a)
    random.shuffle(b)
    print(a)
    print(b)
