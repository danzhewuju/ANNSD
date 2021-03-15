#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/4 15:57
# @Author  : Alex
# @Site    : 
# @File    : positive_data.py
# @Software: PyCharm

"""
对于一些只有阳性批次的脑电信号进行处理，值得注意的是处理前需要进行手动的切分；
本文件代码专门用于处理部分的阳线的数据
处理步骤：1. 读取文件
         2. 文件切分
         3. 信道排序
         4. 滤波重采样
         5. 随机切片
         6. 存储生成路径信息
         7. 文件写回
"""
import json

import pandas
from tqdm import tqdm
import sys
sys.path.append('../')

from util.seeg_utils import *


def sampling_raw_data(new_data_positive="./new_data_positive.csv", window=(2, 16), config_path="./config/config.json",
                      save_dir="../data", k=5):
    """

    :param new_data_positive: 需要处理的csv文件路径,需要手动指定生成
    :param config_path: 信道文件信息的存放路径
    :param k: 平均采样倍率的计算，实际的采样的个数计算如下：
                                  count = (duration time)*2*k/(average length)
                                  这样计算的好处是为了使得采样的个数和数据长度相关
    :return:
    """
    data_path = pandas.read_csv(new_data_positive, sep=',')
    config_data = json.load(open(config_path, 'r'))
    for index, row in data_path.iterrows():
        ent_time = row['Pre_Seizure Duration(s)']  # 需要切分的终止时间
        raw_data_path = row['Path']
        file_name = row['File'].split('.')[0]
        patient = row['patient']
        save_dir_patient = os.path.join(save_dir, "{}/pre_seizure".format(patient))
        data = read_edf_raw(raw_data_path)  # 读取文件
        data_cut = get_duration_raw_data(data, 0, int(ent_time))  # 数据被切分为一个片段，这是一个有效的片段
        channel_path = config_data["{}_data_path".format(patient)]['data_channel_path']
        # 开始进行数据预处理

        data = re_sampling(data_cut, fz=500)
        channel_name = pandas.read_csv(channel_path)
        channels_name = list(channel_name['chan_name'])
        data = select_channel_data_mne(data, channels_name)
        # 开始进行数据数据切分
        gen_time = []
        count = int((int(ent_time)*2*k)/(window[0]+window[1]))

        for i in range(count):
            start = np.random.randint(0, ent_time)  # 计算起始时间
            dur = np.random.randint(window[0], window[1])
            end = min(start + dur, ent_time)  # 如果超过了最大的时间，就取最小
            gen_time.append((start, end))
        if not os.path.exists(save_dir_patient):
            os.makedirs(save_dir_patient)  # 递归的构建文件夹
        fz = get_sampling_hz(data)  # 获得该数据的采样率
        for s, e in tqdm(gen_time):
            name = str(uuid.uuid1()) + "-{}-{}_{}.npy".format(file_name, s, e)
            path = os.path.join(save_dir_patient, name)
            s = int(s * fz)
            e = int(e * fz)
            data_slicing, _ = data[:, s:e]
            np.save(path, data_slicing)
        print("Data processing finished!")


if __name__ == '__main__':
    sampling_raw_data()
