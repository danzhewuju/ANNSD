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
import  json


def train_data_split(patient_name = "BDP", data_info_path = "../preprocess/data_info.csv"):
    '''
    : function 用于数据集的划分
    :param patient_name: 指定病人的数据用于测试
    :param data_info_path: 整体的病人信息的汇总
    :return:
    '''
