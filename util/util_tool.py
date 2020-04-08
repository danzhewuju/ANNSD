#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:58
# @Author  : Alex
# @Site    : 
# @File    : util_tool.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import json
import random


def train_data_split(patient_name="BDP", data_info_path="../preprocess/data_info.csv"):
    '''
    : function 用于数据集的划分
    :param patient_name: 指定病人的数据用于测试
    :param data_info_path: 整体的病人信息的汇总
    :return:
    '''

    data = pd.read_csv(data_info_path)
    ratio = 0.7
    val = {'path': [], 'label': [], 'patient': []}
    paths = data['path']
    label = data['label']
    patient = data['patient']
    for i in range(len(paths)):
        if patient[i] == patient_name:
            val['path'] += paths[i]
            paths.pop(i)
            val['label'] += label[i]
            label.pop(i)
            val['patient'] += patient[i]
            patient.pop(i)
    print(len(paths))
    time_seed = random.randint(0, 100)
    random.seed(time_seed)
    random.shuffle(paths)
    random.seed(time_seed)
    random.shuffle(label)
    random.seed(time_seed)
    random.shuffle(patient)
    train_num = int(ratio * len(paths))
    train_data = {'path': paths[:train_num], 'label': label[:train_num], 'patient': patient[:train_num]}
    test_data = {'path': paths[train_num:], 'label': label[train_num:], 'patient': patient[train_num:]}
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    val_data = pd.DataFrame(val)
    train_path = './train.csv'
    test_path = './test.csv'
    val_path = './val.csv'
    with open(train_path, 'w') as f:
        train_data.to_csv(f)
    with open(test_path, 'w') as f:
        test_data.to_csv(f)
    with open(val_path, 'w') as f:
        val_data.to_csv(f)
    print("数据划分完成")


train_data_split()
