#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/31 19:57
# @Author  : Alex
# @Site    : 
# @File    : generatedata.py
# @Software: PyCharm


from util.seeg_utils import *
import numpy as npo
import os
import pandas as pd
from tqdm import tqdm
import json


class GenData:

    def generate_data(self, duration, wins=(2, 16), k=3000):
        '''

        :param duration:  输入文件的持续时长，是文件的长度
        :param wins：  随机化窗口的大小的范围（2, 16）：随机化的窗口的取值范围为：2-15
        :param k: 生成的数据的个数
        :return:
        '''
        res = []
        for i in range(k):
            start = np.random.randint(0, duration)  # 计算起始时间
            dur = np.random.randint(wins[0], wins[1])
            end = min(start + dur, duration)
            res.append((start, end))
        return res

    def input_data(self, data_path, channel_path):
        '''

        :param data_path: 数据的输入处理
        :return:
        '''
        data = read_raw(data_path)
        data = re_sampling(data, fz=500)  # 对于数据进行重采样
        channel_name = pd.read_csv(channel_path)
        channels_name = list(channel_name['chan_name'])
        self.data = select_channel_data_mne(data, channels_name)  # 将数据的信道进行重组
        self.duration = get_recorder_time(self.data)

    def save_data(self, save_dir):
        # 需要将切片的数据进行保存
        '''
        :return:
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 递归的构建文件夹
        fz = get_sampling_hz(self.data)  # 获得该数据的采样率
        for s, e in tqdm(self.gen_time):
            name = str(uuid.uuid1()) + "_pre-{}_{}.npy".format(s, e)
            path = os.path.join(save_dir, name)
            s = int(s * fz)
            e = int(e * fz)
            data_slicing, _ = self.data[:, s:e]
            np.save(path, data_slicing)
        print("Data processing finished!")

    def __init__(self, data_path, channel_path, save_dir, wins=(2, 16), k=3000):
        self.input_data(data_path, channel_path)
        self.gen_time = self.generate_data(self.duration, wins, k)  # 生成数据的其实时间 时间的长度并相同
        self.save_data(save_dir)


def process_data(config_path="./config/config.json"):
    '''

    :param config_path:  配置文件的存储路径
    :return:
    '''
    K = 3000
    # 每一个病人以及每一个状态的选取的采样的片段的个数
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(config)
    for id, info in config.items():
        print("处理病人：{}相关数据".format(id.split("_")[0]))
        channel_path = info["data_channel_path"]
        non_raw_data_path = info['non_seizure']['raw_data_path']
        non_save_dir = info['non_seizure']['save_dir']
        r_k = K // len(non_raw_data_path)
        for p in non_raw_data_path:
            gen_data = GenData(p, channel_path, non_save_dir, (2, 16), r_k)
        print("正常睡眠数据处理完成")
        pre_raw_data_path = info['pre_seizure']['raw_data_path']
        pre_save_dir = info['pre_seizure']['save_dir']
        r_k = K // len(pre_raw_data_path)
        for p in pre_raw_data_path:
            gen_data = GenData(p, channel_path, pre_save_dir, (2, 16), r_k)
        print("癫痫发作前睡眠数据处理完成")


def get_data_info(path_dir="../data/"):
    '''
    :function: 主要生成数据相关的统计信息，用于标签的训练 
    :param path_dir:
    :return:
    '''


if __name__ == '__main__':
    process_data()  # 数据的生成, 主要是用与生成时序数据
