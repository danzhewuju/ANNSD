'''
本文件主要是用于vmaml模型预测结果的voting处理过程，为了是结果更具有健壮性
'''
import numpy as np
import argparse
import sys

sys.path.append('../')
from util.util_file import IndicatorCalculation, calculation_result_standard_deviation
from util.util_file import LogRecord


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vl', '--vote_length', type=int, default=15, help="Length of voting time")
    arg = parser.parse_args()
    vote_length = arg.vote_length

    patients = ['BDP', 'SYF', 'WSH', 'ZK', 'LK']
    for p in patients:
        path = './vmodel_prediction/{}_val_prediction.pkl'
        tmp = path.format(p)
        vmaml_voting = VmamlVoting(tmp, vote_length, patient=p)
