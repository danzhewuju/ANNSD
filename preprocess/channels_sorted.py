#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 15:56
# @Author  : Alex
# @Site    : 
# @File    : channels_sorted.py
# @Software: PyCharm

"""
批量处理信道排序

"""
import collections
import os

import pandas as pd
from tqdm import tqdm

from util.seeg_utils import least_traversal, retrieve_chs_from_mat


def channels_sorted(file_csv, save_dir):
    """
    批量计算所有的信道文件，并且根据信道信息进行重排序

    :param file_csv: 配置的文件，包含所有mat文件的csv表
    :param save_dir: 存储的文件夹
    :return:
    """
    data_info = pd.read_csv(file_csv)
    for index, row in tqdm(data_info.iterrows()):
        path = row['path']
        # 根据路径来获取病人名字
        patient_name = os.path.basename(path).split('.')[0]
        # 目录的保存路径
        save_path = os.path.join(save_dir, patient_name + "_seq.csv")

        channel_info = collections.defaultdict(list)
        data = retrieve_chs_from_mat(path)
        # 需要增加一个挑选坏信道的模块

        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------

        _, channels_list = least_traversal(data)
        channel_info['ID'] = list(range(len(channels_list)))
        channel_info['chan_name'] = channels_list
        DataFrame = pd.DataFrame(channel_info)
        DataFrame.to_csv(save_path, index=False)
    print("All channel files has been processed! ")


if __name__ == '__main__':
    FILE_CSV = "/data/yh/dataset/channels_positive/allMatInfo.csv"
    SAVE_DIR = "/data/yh/dataset/channels_positive/channel_sorted"
    channels_sorted(FILE_CSV, SAVE_DIR)
