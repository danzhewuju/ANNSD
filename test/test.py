# print("Hello")
# path = "/home/yh/yh/dataset/positiveDataAvailable/BDP/BDP{}.edf"
# # for i in range(1, 6):
# i = 1
# new_path = path.format(i)
# data = read_edf_raw(new_path)
# ano = read_annotations(data)
# print(ano)
# print(ano['onset'])
#
# i = 2
# new_path = path.format(i)
# data = read_edf_raw(new_path)
# ano = read_annotations(data)
# print(ano)
# print(ano['onset'])
#
# i = 3
# new_path = path.format(i)
# data = read_edf_raw(new_path)
# ano = read_annotations(data)
# print(ano)
# print(ano['onset'])
#
# i = 4
# new_path = path.format(i)
# data = read_edf_raw(new_path)
# ano = read_annotations(data)
# print(ano)
# print(ano['onset'])
#
# i = 5
# new_path = path.format(i)
# data = read_edf_raw(new_path)
# ano = read_annotations(data)
# print(ano)
# print(ano['onset'])
#
# i = 6
# new_path = path.format(i)
# data = read_edf_raw(new_path)
# ano = read_annotations(data)
# print(ano)
# print(ano['onset'])


# # path = "/data/yh/Python/SEEG-Timing/log/raw_data_info.csv"
# # data = pd.read_csv(path)
# # for index, row in data.iterrows():
# #     print(row['Pre_Seizure Duration(s)'])
#
# bad_path = "/data/yh/dataset/channels_positive/badChannels.csv"
# data = pd.read_csv(bad_path, sep='\t')
# print(data.head())

# from util.seeg_utils import read_edf_raw, get_channels_names
# path = "/home/yh/yh/dataset/positiveDataAvailable/WJ/WJ1.edf"
# data = read_edf_raw(path)
# print(get_channels_names(data))
# print(len(get_channels_names(data)))
import pandas

from util.seeg_utils import *

# a = {'a': [1, 2, 3], 'b': [2, 3, 4]}
# data = pd.DataFrame(a)
# # data['ACC'] = 1
#
# for index, row in data.iterrows():
#     row[''] = 1
#
# print(get_sampling_hz(read_edf_raw("/home/yh/yh/dataset/positiveDataAvailable/KLJ/KLJ2.2.edf")))

a = {"a": [1]}
data = pandas.DataFrame(a)
b = {'a': [2]}
data_b = pandas.DataFrame(b)
data = data.append(data_b, ignore_index=True)
print(data)
