#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 16:42
# @Author  : Alex
# @Site    : 
# @File    : draw.py
# @Software: PyCharm

"""
主要是用于数据图形的可视化，但是不针对某一种特别的数据，为了画出数据的示意图

包含  1.单信道的示意图
     2. 多个信道的示意图

"""

import os

import matplotlib.pyplot as plt
import numpy as np

from util.seeg_utils import read_edf_raw, read_raw, get_sampling_hz


class Draw:

    def read_raw_data(self, path):
        fix = path.split('.')[-1]  # 后缀
        if fix == "npy":
            data = np.load(path)
        elif fix == "fif":
            data = read_raw(path)
        elif fix == "edf":
            data = read_edf_raw(path)
        else:
            pass
        return data

    def read_channel_duration_data(self, channel, start_time, end_time, resampling=500):
        """

        :param channel: The index of channel
        :param start_time: The segment you want to stating.
        :param end_time:  The segment you want to ending.
        :return:
        """
        fix = self.path.split(".")[-1]
        if fix == "npy":
            if (end_time - start_time) * resampling > self.data.shape[-1]:
                assert "The length of segment is too long. "
            data = self.data[channel][start_time * resampling:end_time * resampling]
        else:
            sampling = get_sampling_hz(self.data)
            data, _ = self.data[channel, start_time * sampling:end_time * sampling]
        return data

    def draw_signal_wave(self, channel, start_time, end_time, save_path):
        base_dir = os.path.dirname(save_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print("Dir is not existed, Creating a dir for your save path.")
        data = self.read_channel_duration_data(channel, start_time, end_time)
        data = data.flatten()
        plt.figure(figsize=(int((end_time - start_time) * 1.5), max(1, int((end_time - start_time) / 5))))
        plt.plot(range(len(data)), data)
        plt.yticks([])
        plt.xticks([])
        plt.savefig(save_path)
        plt.show()
        return

    def __init__(self, path):
        self.data = self.read_raw_data(path)  # Read data.
        self.path = path


if __name__ == '__main__':
    # 1. 癫痫发作前
    draw = Draw(path="/home/yh/yh/dataset/raw_data/BDP/BDP_Pre_seizure/BDP_SZ1_pre_seizure_raw.fif")
    draw.draw_signal_wave(58, 25, 45, save_path="./plot/signal_25-45.pdf")

    # 2. 正常睡眠

    draw = Draw(path="/home/yh/yh/dataset/raw_data/BDP/BDP_SLEEP/BDP_Sleep_raw.fif")
    draw.draw_signal_wave(58, 20, 40, save_path="./plot/sleep_20-40.pdf")
