import argparse
import collections
import os
import re
import sys

import pandas

sys.path.append('../')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from util.seeg_utils import read_edf_raw, read_annotations, get_recorder_time, read_raw
from util.util_file import IndicatorCalculation, LogRecord


class Information:

    @staticmethod
    def read_raw_data(path):
        fix = path.split('.')[-1]  # 后缀
        data = None
        if fix == "npy":
            data = np.load(path)
        elif fix == "fif":
            data = read_raw(path)
        elif fix == "edf":
            data = read_edf_raw(path)
        else:
            pass
        return data

    def draw_plt_bar(self, data, x_laebl, y_label):
        """

        :param data:  字典特征的数据
        :param x_laebl: 横坐标
        :param y_label:  纵坐标
        :return:
        """
        data = dict(sorted(data.items(), key=lambda x: x[0]))
        x, y = list(data.keys()), list(data.values())
        plt.figure()
        plt.bar(x, y)
        plt.xlabel(x_laebl)
        plt.ylabel(y_label)
        plt.show()
        plt.close()

    def calculate_attention_info(self, path_dir='../log/attention'):
        names = os.listdir(path_dir)
        res = collections.defaultdict(int)
        for n in names:
            tmp = os.path.basename(n).split('-')[-1]
            new_time = [int(x) for x in re.findall(r'\d+', tmp)]
            start_index = new_time[0]
            path = os.path.join(path_dir, n)
            data = np.load(path)
            sum_data = np.sum(data, axis=0)
            index = np.argmax(sum_data, axis=0)
            point = start_index + index
            res[point] += 1
        self.draw_plt_bar(dict(res), 'Time(s)', 'Count')
        return dict(res)

    def create_attention_csv(self, file_name='../preprocess/attention_BDP.csv', time=(340, 400),  # 选出符合条件范围的测试用例
                             test_file='../preprocess/test_BDP.csv'):
        test_data = pd.read_csv(test_file)
        path = test_data['path'].tolist()
        label = test_data['label'].tolist()
        patient = test_data['patient'].tolist()
        count = 1500  # 分析单一文件，一个文件采样了1500个
        path_att, label_att, patient_att = [], [], []
        for i in range(count):
            p = path[i]
            tmp = os.path.basename(p).split('-')[-1]
            new_time = [int(x) for x in re.findall(r'\d+', tmp)]
            start_time, end_time = new_time[0], new_time[1]
            if start_time >= time[0] and end_time <= time[1]:
                path_att.append(path[i])
                label_att.append(label[i])
                patient_att.append(patient[i])
        dataframe = {'path': path_att, 'label': label_att, 'patient': patient_att}
        DataFrame = pd.DataFrame(dataframe)
        DataFrame.to_csv(file_name)
        print("Processing {} data!".format(len(path_att)))

    def calculation_prediction(self, file_path='../log/prediction_result.csv', time_gap=60):
        data = pd.read_csv(file_path)
        data.sort_values(by='id', ascending=True, inplace=True)
        id_list = data['id'].tolist()
        grand_true = data['grand true'].tolist()
        prediction = data['prediction'].tolist()

        prediction_true = [1 if grand_true[i] == prediction[i] else 0 for i in range(len(id_list))]
        count = (0, 1500)
        res = collections.defaultdict(list)
        for i in range(len(id_list)):
            if count[0] < i < count[1] and grand_true[i] == 1:
                id = id_list[i]
                tmp = os.path.basename(id).split('-')[-1]
                new_time = [int(x) for x in re.findall(r'\d+', tmp)]
                start_time = new_time[0]
                res[start_time // time_gap + 1] += [1] if prediction_true[i] == 1 else [0]
        acc_info = {}
        for k, v in res.items():
            acc_info[k] = sum(v) / len(v)
        self.draw_plt_bar(acc_info, x_laebl='Time(min)', y_label='Accuracy')
        return acc_info

    @staticmethod
    def calculation_index(file_path, log_file, patient):
        """
        # 用于计算指定指定文件预测的各种指标，该文件是预测后的文件
        :param patient: 病人信息
        :param log_file: 日志文件
        :param file_path 文件的路径，分析病人的指标
        """
        data = pd.read_csv(file_path)
        data.sort_values(by='id', ascending=True, inplace=True)
        id_list = data['id'].tolist()
        ground_true = data['ground truth'].tolist()
        prediction = data['prediction'].tolist()
        cal = IndicatorCalculation(prediction, ground_true)
        Accuracy, Precision, Recall, F1score, AUC = cal.get_accuracy(), cal.get_precision(), cal.get_recall(), cal.get_f1score(), cal.get_auc()
        result = "Patient:{}|DataSize:{}|Acuracy:{:.6f}|Precision:{:.6f}|Recall:{:.6f}|F1score:{:.6f}| AUC:{:.6f}".format(
            patient, len(id_list), Accuracy, Precision, Recall, F1score, AUC)
        LogRecord.write_log(result, log_file)
        return

    @staticmethod
    def raw_data_info(dir_path="/home/yh/yh/dataset/positiveDataAvailable", save_file="../log/raw_data_info.csv"):
        """
        原始数据统计模块，主要是统计数据中是否存在医生标注的癫痫发作信息
        :param dir_path: 数据解压的文件夹
        :param save_file 存储的文件信息
        :return:
        """
        DataFrame = collections.defaultdict(list)
        name_list = os.listdir(dir_path)
        for n in name_list:
            tmp_dir = os.path.join(dir_path, n)
            file_name_list = os.listdir(tmp_dir)
            for fname in file_name_list:
                full_path = os.path.join(tmp_dir, fname)
                data = Information.read_raw_data(full_path)
                if not data:
                    continue
                annotations = read_annotations(data)
                seizure_time = int(annotations['onset'][-1]) if len(annotations['onset']) > 3 else -1
                DataFrame['patient'].append(n)
                DataFrame['File'].append(fname)
                DataFrame['Path'].append(full_path)
                DataFrame['All Duration(s)'].append(int(get_recorder_time(data)))
                DataFrame['Seizure(s)'].append(seizure_time)
                DataFrame['Pre_Seizure Duration(s)'].append(max(seizure_time - 30, 0))

        dataFrame = pd.DataFrame(DataFrame)
        try:
            dataFrame.to_csv(save_file, index=False)
        except IOError:
            print("File error!")
        return

    @staticmethod
    def window_accuracy(file_path, save_path):
        """
        :function 研究窗口大小对于准确率的影响
        :param file_path: 文件路径
        :return:
        """
        data = pandas.read_csv(file_path)
        windows_count = {}
        for index, row in data.iterrows():
            id_str = row['id']
            tmp_str = id_str.split('-')[-1].split('_')
            start_time, end_time = int(tmp_str[0]), int(tmp_str[1])
            windows_size = end_time - start_time
            ground_truth, prediction = row['grand true'], row['prediction']
            if windows_size not in windows_count.keys():
                tmp = [0, 0]
                if ground_truth == prediction:
                    tmp[0] += 1
                tmp[1] += 1
                windows_count[windows_size] = tmp
            else:
                if ground_truth == prediction:
                    windows_count[windows_size][0] += 1
                windows_count[windows_size][1] += 1
        data_frame = collections.defaultdict(list)
        for w, (count, all) in windows_count.items():
            data_frame['window'].append(w)
            accuracy = format(count / all, '.5f')
            data_frame['accuracy'].append(accuracy)
        DataFrame = pandas.DataFrame(data_frame)
        DataFrame.to_csv(save_path, index=False)
        print("Result has been saved!")
        return


def menu():
    parse = argparse.ArgumentParser()
    parse.add_argument('-cac', '--create_attention_csv', type=bool, default=False)
    parse.add_argument('-cai', '--calculate_attention_info', type=bool, default=True)
    parse.add_argument('-cp', '--calculation_prediction', type=bool, default=True)
    parse.add_argument('-m', '--model', type=str, default='rawInfo', help="Analysis data information ")
    parse.add_argument('-fp', '--file_path', type=str, help="Analysis file path")
    parse.add_argument('-p', '--patient', type=str, help="Patient name")
    parse.add_argument('-lf', '--log_file', type=str, help="Log file", default='../log/log.txt')
    args = parse.parse_args()
    info = Information()

    cac, cai, cp = args.create_attention_csv, args.calculate_attention_info, args.calculation_prediction
    model = args.model

    if model == 'analysis':
        """
        日志分析模块，分析给定预测结果的日志文件，再重新计算相关的指标
        """
        file_path = args.file_path
        log_file = args.log_file
        patient = args.patient
        Information.calculation_index(file_path, log_file, patient)
    elif model == 'windows_accuracy':
        """
        研究窗口的大小和准确率的关系
        """
        parse.add_argument('-sp', '--save_path', type=str, help="Log file", default='../log/all_windows_accuracy.csv')
        args = parse.parse_args()
        save_path = args.save_path
        file_path = args.file_path
        Information.window_accuracy(file_path, save_path)

    elif model == "attention":
        """
        attention 分析计算模块
        主要是分析模型的attention相关信息；
        流程： 1 -> 选择符合时间段的数据片段； 2 -> 计算相关的attention值;  3 -> 计算最后的预测结果；    
        """
        if cac:
            info.create_attention_csv()  # 选出符合条件范围的测试用例,选出指定时间段的模型的准确率
        if cai:
            print(info.calculate_attention_info())  # 计算attention的信息
        if cp:
            print(info.calculation_prediction())  # 计算预测的结果
    elif model == "rawInfo":
        """
        原始数据统计模块，主要是统计数据中是否存在医生标注的癫痫发作信息
        """

        parse.add_argument('-dp', '--dir_path', type=str, help="dir path",
                           default="/home/yh/yh/dataset/positiveDataAvailable")
        parse.add_argument('-sp', '--save_file', type=str, help="Save file", default="../log/raw_data_info.csv")
        args = parse.parse_args()
        dir_path, save_path = args.dir_path, args.save_file

        Information.raw_data_info(dir_path, save_path)
    else:
        pass


if __name__ == '__main__':
    menu()

# info.create_attention_csv()   # 选出符合条件范围的测试用例
# print(info.calculate_attention_info()) # 计算attention的信息
# print(info.calculation_prediction())  # 计算预测的结果
