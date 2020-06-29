#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/31 20:20
# @Author  : Alex
# @Site    : 
# @File    : test.py
# @Software: PyCharm
# from data.generatedata import generate_data
import random

from torch.autograd import Variable
from util.seeg_utils import *
from tqdm import tqdm
import time
import os

import torch
import matplotlib.pyplot as plt
import collections
import numpy as np
from util.seeg_utils import draw_seeg_picture
import argparse
from util.run_util import get_gpu_used
import torch.nn as nn
import re


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def test_1():
    Tensor = torch.FloatTensor
    valid = Variable(Tensor(64, 1).fill_(1.0), requires_grad=False)
    print(valid)


def test_2():
    def draw_loss_plt(*args, **kwargs):
        # 画出train或者test过程中的loss曲线
        loss_l, loss_d, loss_t, acc = kwargs['loss_l'], kwargs['loss_d'], kwargs['loss_t'], kwargs['acc']
        plot_save_path = kwargs['save_path']
        model_info = kwargs['model_info']
        dir_name = os.path.dirname(plot_save_path)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Create dir {}".format(dir_name))
        plt.figure()
        plt.xlabel('Step')
        plt.ylabel('Loss/Accuracy')
        plt.title(model_info)
        x = range(len(acc))
        plt.plot(x, loss_l, label="Loss of label classifier")
        plt.plot(x, loss_d, label="Loss of domain discriminator")
        plt.plot(x, loss_t, label='Total loss')
        plt.plot(x, acc, label='Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(plot_save_path)
        plt.show()
        print("The Picture has been saved.")

    loss_l = torch.randn(100)
    loss_d = torch.randn(100)
    loss_t = (loss_l + loss_d) / 2
    acc = torch.randn(100)
    info = {'loss_l': loss_l, 'loss_d': loss_d, 'loss_t': loss_t, 'acc': acc, 'save_path': './draw/train_loss_pair.png',
            'model_info': "training information"}
    draw_loss_plt(**info)
    info = {'loss_l': loss_l, 'loss_d': loss_d, 'loss_t': loss_t, 'acc': acc, 'save_path': './draw/test_loss.png',
            'model_info': "test information"}
    draw_loss_plt(**info)


def test_3():
    n = 5
    m = 300
    with tqdm(total=n * m) as pbar:
        for i1 in tqdm(range(n)):
            for i2 in tqdm(range(m)):
                # do something, e.g. sleep
                time.sleep(0.01)
                pbar.update(1)


def test_4():
    code_x2_order = list(range(16))
    pre_half, behind_half = code_x2_order[:len(code_x2_order) // 2], code_x2_order[len(code_x2_order) // 2:]
    random.shuffle(behind_half)  # 只打乱一半的数据
    code_x2_order = pre_half + behind_half

    print(code_x2_order)


def test_5():
    path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/raw_data/BDP/BDP_Pre_seizure/BDP_SZ2_pre_seizure_raw.fif"
    data = read_raw(path)
    time = get_recorder_time(data)
    print(time)


def test_6():
    path = "../data/BDP/pre_seizure/fb59ab43-789b-11ea-8178-e0d55e6ff654_pre-420_430.npy"

    data = np.load(path)
    draw_seeg_picture(data)
    # attention_path = "../log/attention.npy"
    # data_attention = np.load(attention_path)
    # # data_attention = np.random.randint(0, 10, (2, 3, 3))
    # att = data_attention.sum(axis=0) / 8
    # plt.figure()
    # plt.imshow(att, cmap='hot')
    # plt.xlabel('Time(s)-Query')
    # plt.ylabel('Time(s)-Key')
    # plt.show()
    # plt.close()
    # print(att)
    # att = att.sum(axis=0)
    # att = softmax(att)
    # print(att)
    # plt.figure()
    # plt.imshow(att, cmap='hot')
    # plt.xlabel('Time(s)')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    test_5()
    # a = np.random.randn(2, 3)
    # b = a.sum(axis=0)
    #
    # print(a)
    # print(b)
