import numpy as np
import os
import collections
import pandas as pd
import re
from matplotlib import pyplot as plt


class Information:
    def draw_plt_bar(self, data, x_laebl, y_label):
        '''

        :param data:  字典特征的数据
        :param x_laebl: 横坐标
        :param y_label:  纵坐标
        :return:
        '''
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

    def create_attention_csv(self, file_name='../preprocess/attention_BDP.csv', time=(340, 400),
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


if __name__ == '__main__':
    info = Information()
    # info.create_attention_csv()   # 选出符合条件范围的测试用例
    # print(info.calculate_attention_info())
    print(info.calculation_prediction())
