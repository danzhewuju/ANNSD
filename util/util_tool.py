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
from torch.utils.data import Dataset
from util.util_file import matrix_normalization
import torch
import torch.nn.utils.rnn as rnn_utils


def train_data_split(patient_name="BDP", data_info_path="../preprocess/data_info.csv"):
    '''
    : function 用于数据集的划分
    :param patient_name: 指定病人的数据用于测试
    :param data_info_path: 整体的病人信息的汇总
    :return:
    '''

    data = pd.read_csv(data_info_path, sep=',')
    ratio = 0.7
    val = {'path': [], 'label': [], 'patient': []}
    paths = data['path']
    label = data['label']
    patient = data['patient']
    for i in range(len(paths)):
        if patient[i] == patient_name:
            val['path'] += [paths[i]]
            paths.pop(i)
            val['label'] += [label[i]]
            label.pop(i)
            val['patient'] += [patient[i]]
            patient.pop(i)
    # 对于数据需要进行乱序的处理
    paths = paths.tolist()
    label = label.tolist()
    patient = patient.tolist()
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
    train_path = './train_{}.csv'.format(patient_name)
    test_path = './test_{}.csv'.format(patient_name)
    val_path = './val_{}.csv'.format(patient_name)
    with open(train_path, 'w') as f:
        train_data.to_csv(f, index=None)
    with open(test_path, 'w') as f:
        test_data.to_csv(f, index=None)
    with open(val_path, 'w') as f:
        val_data.to_csv(f, index=None)
    print("数据划分完成")


def collate_fn(data):  #
    '''
    用于自己构造时序数据，包含数据对齐以及数据长度
    :param data: torch dataloader 的返回形式
    :return:
    '''
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

class Data_info():
    def __init__(self, path_train):
        dict_label = {"pre_seizure": 1, "non_seizure": 0}
        data = pd.read_csv(path_train)
        data_path = data['path'].tolist()
        label = data['label'].tolist()
        self.data = []
        for i in range(len(data_path)):
            self.data.append((data_path[i], dict_label[label[i]]))
        self.data_length = len(self.data)

    def next_batch_data(self, batch_size):  # 用于返回一个batch的数据
        N = self.data_length
        start = 0
        end = batch_size
        while end < N:
            yield self.data[start:batch_size]
            start = end
            end += batch_size
            if end >= N:
                start = 0
                end = batch_size


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data = np.load(fn)
        # data = self.transform(data)
        result = matrix_normalization(data, (100, 500))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        # result = trans_data(vae_model, result)
        return result, label

    def __len__(self):
        return len(self.imgs)



