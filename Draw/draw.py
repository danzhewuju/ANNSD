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

from util.seeg_utils import read_edf_raw, read_raw, get_sampling_hz, get_channels_names
import mne
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from matplotlib.ticker import MultipleLocator


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
        """
        :functions: 画出某一信道的某段时间的示意波形图
        :param channel:  信道编号
        :param start_time: 开始时间
        :param end_time:  结束时间
        :param save_path:  保存目录
        :return:
        """
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

    def draw_multi_wave(self, channels: list, start_time: int, end_time: int, save_path: str):
        """
        :function: 将多行的脑电画在同一幅画中
        :param channels:    信道的列表
        :param start_time:  开始时间
        :param end_time: 结束时间
        :param save_path: 保存目录
        :return:
        """
        # 从edf文件读取脑电数据
        raw = self.read_raw_data(self.path)
        channel_names = get_channels_names(raw)
        channel_names_str_list = [channel_names[t] for t in channels]
        # chans = ["POL G9", "POL G5", "POL G4", "POL G2", "POL G3", "POL H3", "POL G7", "POL G8", "POL G6", "POL H1"]
        selection = raw.crop(start_time, end_time)
        selection = selection.pick_channels(channel_names_str_list)

        sl = selection[:, :]  # 抽取为array格式
        offset = np.arange(0, 10 * 0.002, 0.002)
        x = sl[1]  # x轴数据
        y = sl[0].T + offset  # y轴数据

        # ylabel = ['$G2$', '$G3$', '$G4$', '$G5$', '$G6$', '$G7$', '$G8$', '$G9$', '$H1$', '$H3$']  # y轴刻度的名称
        fig = plt.figure()
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        ax.axis["left"].set_axisline_style("->", size=1.5)  # 设置y轴样式为->箭头
        ax.axis["bottom"].set_axisline_style("->", size=1.5)  # 设置x轴样式为->箭头
        ax.axis["top"].set_visible(False)  # 隐藏上面的轴
        ax.axis["right"].set_visible(False)  # 隐藏右侧的轴
        x_major_locator = MultipleLocator(1)  # 设置刻度间距为1
        ax.xaxis.set_major_locator(x_major_locator)
        # plt.yticks(offset.tolist(), ylabel)  # 修改y轴刻度的名称
        plt.xlabel("Time(s)")
        plt.ylabel("Channels")
        # plt.axvline(5, linestyle="dotted", color='k')  # 在x=5的地方画垂直点状线
        # plt.text(5.17, 0.02, "Seizure")  # 在x=5.17,y=0.02处写上Seizure字样
        plt.plot(x, y, '-k', linewidth=0.5)  # 设置线条颜色、宽度

        plt.show()

    def __init__(self, path):
        self.data = self.read_raw_data(path)  # Read data.
        self.path = path


if __name__ == '__main__':
    # 1. 癫痫发作前
    # draw = Draw(path="/home/yh/yh/dataset/raw_data/BDP/BDP_Pre_seizure/BDP_SZ1_pre_seizure_raw.fif")
    # draw.draw_signal_wave(58, 25, 45, save_path="./plot/signal_25-45.pdf")

    # 2. 正常睡眠

    # draw = Draw(path="/home/yh/yh/dataset/raw_data/BDP/BDP_SLEEP/BDP_Sleep_raw.fif")
    # draw.draw_signal_wave(58, 20, 40, save_path="./plot/sleep_20-40.pdf")

    # 3. 将多条脑电信号保存在一幅图中
    draw = Draw(path="/home/yh/yh/dataset/raw_data/BDP/BDP_SLEEP/BDP_Sleep_raw.fif")
    draw.draw_multi_wave([0, 1, 2, 3, 4], 15, 25, "./plot/mutil_wave_preseizure.png")
