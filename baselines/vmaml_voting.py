'''
本文件主要是用于vmaml模型预测结果的voting处理过程，为了是结果更具有健壮性
'''
import numpy as np
import argparse
import sys

sys.path.append('../')
from util.util_file import IndicatorCalculation, calculation_result_standard_deviation
from util.util_file import LogRecord
import pandas as pd


class VmamlVoting:
    def __init__(self, path, vote_length=16, ration=0.6, patient=None):
        '''

        :param path: 分析文件的位置
        :param time_group: 一个成组的时间长度
        :param ration: 达到的比率
        '''
        data = np.load(path, allow_pickle=True)
        print(len(data))
        data = sorted(data.items(), key=lambda x: (x[1]['ground truth'], x[0]))
        grand_truth, prediction = [], []
        for d in data:
            tmp = d[1]
            grand_truth.append(tmp['ground truth'])
            prediction.append(tmp['prediction'])
        # 计算数据原始的准确率
        cal = IndicatorCalculation(prediction, grand_truth)
        # result = "Encoder:{}|Label classifier {}|Patient {}|Data size:{}| test loss:{:.6f}| Accuracy:{:.5f} | Precision:" \
        #          "{:.5f}| Recall:{:.5f}| F1score:{:.5f}| AUC:{:.5f}".format(
        #     self.encoder_name, self.label_classifier_name, self.patient, len(acc), loss_avg, res['accuracy'],
        #     res['precision'], res['recall'], res['f1score'], res['auc'])
        accuracy, precision, recall, f1score = cal.get_accuracy(), cal.get_recall(), cal.get_precision(), cal.get_f1score()
        print(
            "origin_data:Baselines:|Accuracy:{}|Recall:{}|Precision:{}|F1score:{}|AUC:{}".format(accuracy,
                                                                                                 precision,
                                                                                                 recall,
                                                                                                 f1score,
                                                                                                 cal.get_auc()))

        # 计算经过了voting数据的准确率
        grand_truth_g, prediction_g = [], []
        count = 0
        k = vote_length // 2
        while count + k < len(data):
            tmp_grand_truth = sum(grand_truth[count:count + k])
            tmp_prediction = sum(prediction[count:count + k])
            if tmp_grand_truth == k:  # 为阳性
                grand_truth_g.append(1)
                if tmp_prediction > k * ration:
                    prediction_g.append(1)
                else:
                    prediction_g.append(0)
            elif tmp_grand_truth == 0:
                grand_truth_g.append(0)
                if tmp_prediction < k * (1 - ration):
                    prediction_g.append(0)
                else:
                    prediction_g.append(1)
            count += 1
        cal = IndicatorCalculation(prediction_g, grand_truth_g)
        accuracy, a_std = calculation_result_standard_deviation(prediction_g, grand_truth_g, cal.get_accuracy)
        precision, p_std = calculation_result_standard_deviation(prediction_g, grand_truth_g, cal.get_precision)
        recall, r_std = calculation_result_standard_deviation(prediction_g, grand_truth_g, cal.get_recall)
        f1score, f_std = calculation_result_standard_deviation(prediction_g, grand_truth_g, cal.get_f1score)
        auc, auc_std = calculation_result_standard_deviation(prediction_g, grand_truth_g, cal.get_auc)
        # voting 需要进行特殊的方差处理， 进行抽样计算
        result = "Vote Length:{}|Patient:{}|Baselines:VMAML(voting)|Accuracy:{:.6f}|std:{:.6f}|Recall:{:.6f}|std:{:.6f}|Precision:{:.6f}|std:{:.6f}|F1score:{:.6f}|std:{:.6f}|AUC:{:.6f}|std:{:.6f}".format(
            vote_length, patient, accuracy, a_std,
            precision, p_std,
            recall, r_std, f1score, f_std,
            auc, auc_std)
        log_file = '../log/log.txt'
        LogRecord.write_log(result, log_file)  # log 文件的记录
        print("Voting:{}".format(result))


def processing_vmaml_baseline():
    '''

    :return: 不同人的的vmaml voting 的计算
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-vl', '--vote_length', type=int, default=15, help="Length of voting time")
    arg = parser.parse_args()
    vote_length = arg.vote_length

    patients = ['BDP', 'SYF', 'WSH', 'ZK', 'LK']
    for p in patients:
        path = './vmodel_prediction/{}_val_prediction.pkl'
        tmp = path.format(p)
        vmaml_voting = VmamlVoting(tmp, vote_length, patient=p)
    return


def processing_accuracy_time():
    def accuracy_for_time(path_id_index, prediction_file, label, save_file, time_quantum=300):
        '''

        :param path_id_index:  时间序列列表
        :param prediction_file:  预测结果的文件
        :param label: 对于指定类别的数据进行统计分析
        :param time_quantum:    统计一段时间的准确率
        :return:
        '''
        dict_label = {'pre_seizure': 1, 'non_seizure': 0}
        prediction_result = np.load(prediction_file, allow_pickle=True)
        data = pd.read_csv(path_id_index)
        # 过滤掉正常睡眠的信息
        data = data[data['labels'] == label]
        id_list, start_time = data['id'].tolist(), data['start time'].tolist()
        print("id list data size:{}".format(len(id_list)))
        data_time = dict(zip(id_list, start_time))  # 由id构成字典

        # 需要进行分段统计， 第 time_quantum 的时间范围内的结果
        print("data size:{}".format(len(data_time)))
        result = {}  # 用于存储分段的统计信息
        for id_, start_time in data_time.items():
            index = start_time // time_quantum
            if id_ in prediction_result.keys():
                g, p = prediction_result[id_]['ground truth'], prediction_result[id_]['prediction']
            if index not in result.keys():
                result[index] = {'ground truth': [g], 'prediction': [p]}
            else:
                result[index]['ground truth'].append(g)
                result[index]['prediction'].append(p)
        cal = IndicatorCalculation()
        time_, acc, std = [], [], []
        for index_, d in result.items():
            time_.append(index_)
            cal.set_values(d['prediction'], d['ground truth'])
            tmp_acc, tmp_std = calculation_result_standard_deviation(d['prediction'], d['ground truth'],
                                                                     cal.get_accuracy, epoch=2)
            acc.append(tmp_acc)
            std.append(tmp_std)
        # 保留数据小数点
        std = [round(x, 5) for x in std]
        acc = [round(x, 5) for x in acc]
        time_ = [x*time_quantum for x in time_]
        data_dict = {'Time': time_, 'Accuracy': acc, 'Std': std}
        dataFrame = pd.DataFrame(data_dict)
        dataFrame.sort_values(by='Time', inplace=True)
        # 文件的保存
        dataFrame.to_csv(save_file, index=False)
        print("{} file has been created.".format(save_file))
        return True

    patient = 'BDP'
    label = 'pre_seizure'
    path_id_index = './vmodel_prediction/{}_time_index_pre2.csv'.format(patient)
    prediction_file = './vmodel_prediction/{}_prediction_pre2.pkl'.format(patient)
    save_file = './vmodel_prediction/{}_time_accuracy_{}.csv'.format(patient, label)
    accuracy_for_time(path_id_index, prediction_file, label=label, save_file=save_file, time_quantum=180)
    return


if __name__ == '__main__':
    # processing_vmaml_baseline() #不同人的的vmaml voting 的计算
    processing_accuracy_time()  # 分析时间段的准确率
