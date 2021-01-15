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
import sys

from tqdm import tqdm

sys.path.append('../')

from util.seeg_utils import *


def channels_sorted(file_csv, save_dir, bad_channel_path):
    """
    批量计算所有的信道文件，并且根据信道信息进行重排序

    :param bad_channel_path: 需要剔除掉坏道
    :param file_csv: 配置的文件，包含所有mat文件的csv表
    :param save_dir: 存储的文件夹
    :return:
    """

    def mat_id_map_raw_data(mat_channel_name, channelList):
        """
        内部函数返回该名称在原始信道中的名称
        :param mat_channel_name: 需要匹配的信道名称
        :param channelList: 原始数据的信道列表
        :return:
        """
        for cname in channelList:
            new_mat_channel_name = "POL {}".format(mat_channel_name)
            if new_mat_channel_name == cname:
                return cname
            new_mat_channel_name = "EEG {}-Ref".format(mat_channel_name)
            if new_mat_channel_name == cname:
                return cname
            new_mat_channel_name = "POL-{}".format(mat_channel_name)
            if new_mat_channel_name == cname:
                return cname

        return None

    data_info = pd.read_csv(file_csv)
    with tqdm(total=len(data_info)) as pr:
        for index, row in data_info.iterrows():
            path = row['path']
            rawDataPath = row['rawDataPath']
            # 根据路径来获取病人名字
            patient_name = os.path.basename(path).split('.')[0]
            # 目录的保存路径
            save_path = os.path.join(save_dir, patient_name + "_seq.csv")

            channel_info = collections.defaultdict(list)
            data = retrieve_chs_from_mat(path)

            """
            # 由于mat 文件和edf信道的名称并不一致，因此需要增加信道的映射模块
            # 初步思路是从原始文件读取相关模块然后做一个意义匹配的过程
            """
            # --------------------------------------------------------------------------------------------------------------
            # ----------------------名称的映射模块----------------------
            # 读取原始文件
            rawData = read_edf_raw(rawDataPath)
            # 获取原始文件的信道列表
            # 更新整个数据的列表
            rawDataChannels = get_channels_names(rawData)
            new_data = []
            for d in data:
                old_k, old_v = d['name'], d['pos']
                new_k = mat_id_map_raw_data(old_k, rawDataChannels)
                if new_k:
                    new_row = {"name": new_k, "pos": old_v}
                    new_data.append(new_row)
            data = new_data

            # --------------------------------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------------------------------
            # 需要增加一个挑选坏信道的模块
            data_bad_channel = pd.read_csv(bad_channel_path, '\t')
            if patient_name in data_bad_channel:
                bad_channel_patient = data_bad_channel[patient_name].tolist()
                # 去掉空值
                bad_channel_patient = [x for x in bad_channel_patient if x == x]
                # 去掉信道的双引号
                bad_channel_patient = [x[1:-1] for x in bad_channel_patient]
                for i in range(len(data) - 1, -1, -1):
                    d = data[i]
                    key = d['name']
                    if key in bad_channel_patient:
                        # 需要删除坏道
                        del data[i]

                # ----------------------------------------------------------------------------------------------------------

            _, channels_list = least_traversal(data)
            channel_info['ID'] = list(range(len(channels_list)))
            channel_info['chan_name'] = channels_list
            DataFrame = pd.DataFrame(channel_info)
            DataFrame.to_csv(save_path, index=False)
        pr.update(1)
    print("All channel files has been processed! ")


if __name__ == '__main__':
    FILE_CSV = "/data/yh/dataset/channels_positive/allMatInfo.csv"
    SAVE_DIR = "/data/yh/dataset/channels_positive/channel_sorted"
    BAD_CHANNLE_PATH = "/data/yh/dataset/channels_positive/badChannels.csv"
    channels_sorted(FILE_CSV, SAVE_DIR, BAD_CHANNLE_PATH)
