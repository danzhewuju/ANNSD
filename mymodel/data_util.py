import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.append('../')
from util.util_tool import matrix_normalization, collate_fn
import torch


class DataInfo():
    def __init__(self, path_data):
        self.dict_label = {"pre_seizure": 1, "non_seizure": 0}
        self.dict_domain = {'BDP': 0, 'LK': 1, 'SYF': 2, 'WSH': 3, 'ZK': 4}
        data = pd.read_csv(path_data)
        data_path = data['path'].tolist()
        label = data['label'].tolist()
        domain = data['patient'].tolist()
        self.data = []
        for i in range(len(data_path)):
            self.data.append((data_path[i], self.dict_label[label[i]], self.dict_domain[domain[i]]))
        self.data_length = len(self.data)

    def next_batch_data(self, batch_size):  # 用于返回一个batch的数据
        N = self.data_length
        start = 0
        end = batch_size
        while end < N:
            yield self.data[start:batch_size]
            start = end
            end += batch_size
            if end >= N:
                start = 0
                end = batch_size


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label, domain = self.data[index]
        data = np.load(fn)
        # data = self.transform(data)
        result = matrix_normalization(data, (100, 1000))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        # result = trans_data(vae_model, result)
        return result, label, domain

    def __len__(self):
        return len(self.data)


class MyData():
    def __init__(self, path_train, path_test, path_val, batch_size):
        '''

        :param path_train: 训练集数据的路径
        :param path_test: 测试集数据的路径
        :param path_val: 验证集数据的路径
        :param batch_size: 批量的数据
        '''

        self.path_train = path_train
        self.path_test = path_test
        self.path_val = path_val
        self.batch_size = batch_size

    def collate_fn(self, data):  #
        '''
        用于自己构造时序数据，包含数据对齐以及数据长度
        :param data: torch dataloader 的返回形式
        :return:
        '''
        # 主要是用数据的对齐
        data.sort(key=lambda x: x[0].shape[-1], reverse=True)
        max_shape = data[0][0].shape
        labels = []  # 每个数据对应的标签
        length = []  # 记录真实的数目长度
        domains = []
        for i, (d, label, patient) in enumerate(data):
            d_shape = d.shape
            length.append(d.shape[-1])
            if d_shape[-1] < max_shape[-1]:
                tmp_d = np.pad(d, ((0, 0), (0, 0), (0, max_shape[-1] - d_shape[-1])), 'constant')
                data[i] = tmp_d
            else:
                data[i] = d
            labels.append(label)
            domains.append(patient)

        return torch.from_numpy(np.array(data)), torch.tensor(labels), torch.tensor(domains), torch.tensor(length)

    def data_loader(self, mode='train'): # 这里只有两个模式，一个是train/一个是val
        if mode == 'train':
            data_info = DataInfo(self.path_train)
        else:
            data_info = DataInfo(self.path_val)
        dataset = MyDataset(data_info.data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        return dataloader

    def next_batch_test_data(self):
        data_info = DataInfo(self.path_test)
        dataset = MyDataset(next(data_info.next_batch_data(self.batch_size)))
        next_batch_data_loader = DataLoader(dataset, shuffle=True, collate_fn=self.collate_fn)
        yield next_batch_data_loader
