#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/4 15:57
# @Author  : Alex
# @Site    : 
# @File    : positive_data.py
# @Software: PyCharm

"""
对于一些只有阳性批次的脑电信号进行处理，值得注意的是处理前需要进行手动的切分；
处理步骤：1. 读取文件
         2. 文件切分
         3. 信道排序
         4. 滤波重采样
         5. 随机切片
         6. 存储生成路径信息
"""
import json

import pandas
from tqdm import tqdm

from util.seeg_utils import *


def read_data(new_data_positive="./new_data_positive.csv", wins=(2, 16), config_path="./config/config.json",
              save_dir="../data", k=50):
    """

    :param new_data_positive: 需要处理的csv文件路径
    :param config_path: 信道文件信息的存放路径
    :param k: 需要采样的个数
    :return:
    """
    data_path = pandas.read_csv(new_data_positive, sep=',')
    config_data = json.load(open(config_path, 'r'))
    for index, row in data_path.iterrows():
        ent_time = row['Pre_Seizure Duration(s)']  # 需要切分的终止时间
        raw_data_path = row['Path']
        file_name = row['File'].split('.')[0]
        patient = row['patient']
        save_dir_patient = os.path.join(save_dir, "{}/non_seizure".format(patient))
        data = read_edf_raw(raw_data_path)
        data_cut = get_duration_raw_data(data, 0, int(ent_time))  # 数据被切分为一个片段
        channel_path = config_data["{}_data_path".format(patient)]['data_channel_path']
        # 开始进行数据预处理

        data = re_sampling(data_cut, fz=500)
        channel_name = pandas.read_csv(channel_path)
        channels_name = list(channel_name['chan_name'])
        data = select_channel_data_mne(data, channels_name)
        # 开始进行数据数据切分
        gen_time = []

        for i in range(k):
            start = np.random.randint(0, ent_time)  # 计算起始时间
            dur = np.random.randint(wins[0], wins[1])
            end = min(start + dur, ent_time)
            gen_time.append((start, end))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 递归的构建文件夹
        fz = get_sampling_hz(data)  # 获得该数据的采样率
        for s, e in tqdm(gen_time):
            name = str(uuid.uuid1()) + "{}-{}_{}.npy".format(file_name, s, e)
            path = os.path.join(save_dir_patient, name)
            s = int(s * fz)
            e = int(e * fz)
            data_slicing, _ = data[:, s:e]
            np.save(path, data_slicing)
        print("Data processing finished!")


if __name__ == '__main__':
    read_data()
