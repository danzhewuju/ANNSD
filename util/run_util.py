#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 9:33 下午
# @Author  : Alex
# @Site    : 
# @File    : run_util.py
# @Software: PyCharm
# 程序运行的辅助工具
import pynvml


def get_gpu_used(index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    memoryinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return memoryinfo.used


def get_gpu_free(index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    memoryinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return memoryinfo.free
