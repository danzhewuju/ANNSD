# 运行各种baselines的方法
import os
import sys

import torch

from model import clstm, cnnVoting, cnnTransformer, cnnSvm

sys.path.append('../')
import time
from mymodel.data_util import MyData
from torch import nn
from tqdm import tqdm
from util.util_file import IndicatorCalculation


class BaseModel:  # 主要用于选择各种各样的模型
    def __init__(self, gpu, basename, input_size=32, Resampling=500):
        if basename == 'cnnLstm':
            if gpu >= 0:
                self.model = clstm(gpu, input_size, Resampling).cuda(gpu)
            else:
                self.model = clstm(gpu, input_size, Resampling)
        elif basename == 'cnnVoting':  # 采用CNN Voting 的方法
            if gpu >= 0:
                self.model = cnnVoting(gpu, input_size, Resampling).cuda(gpu)
            else:
                self.model = cnnVoting(gpu, input_size, Resampling)
        elif basename == 'cnnTransformer':
            if gpu >= 0:
                self.model = cnnTransformer(gpu, input_size, Resampling).cuda(gpu)
            else:
                self.model = cnnTransformer(gpu, input_size, Resampling)
        elif basename == 'cnnSvm':
            if gpu >= 0:
                self.model = cnnSvm(gpu, input_size, Resampling).cuda(gpu)
            else:
                self.model = cnnSvm(gpu, input_size, Resampling)
        else:
            pass

    def get_model(self):  # 获取构建的模型
        return self.model


class Baselines:
    def __init__(self, patient, epoch=10, bath_size=16, dim=32, lr=0.001, gpu=0, train_path=None, test_path=None,
                 val_path=None,
                 model='train', basename='cnnLstm', few_shot=True, few_show_ratio=0.2, check_point=False, Unbalance=5):
        self.patient = patient  # 患者信息
        self.epoch = epoch  # 迭代次数
        self.batch_size = bath_size  # 批次的大小
        self.dim = dim  # 数据的维度
        self.lr = lr  # 学习率
        self.train_path = train_path  # 训练的数据集
        self.test_path = test_path  # 测试的训练集
        self.val_path = val_path  # 验证集
        self.m = model  # 训练集和测试的选择
        self.basename = basename  # baseline 选择的方法
        self.gpu = gpu  # gpu选择
        self.few_shot = few_shot  # 微调系数开关
        self.few_shot_ratio = few_show_ratio  # 微调系数大小
        self.check_point = check_point  # 断点检查
        self.unbalance = Unbalance  # 非平衡系数
        basemodel = BaseModel(gpu, basename, input_size=self.dim, Resampling=500)  # 构建basemodel方法
        self.model = basemodel.get_model()
        if self.check_point:
            self.load_model()  # 如果是断点训练
            print(" Start checkpoint training")

    def save_mode(self, save_path='../save_model'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_full_path = os.path.join(save_path, 'Baselines_{}_{}.pkl'.format(self.basename, self.patient))
        torch.save(self.model.state_dict(), save_full_path)
        print("Saving Model in {}......".format(save_full_path))
        return

    def load_model(self, model_path='../save_model/Baselines_{}_{}.pkl'):
        model_path = model_path.format(self.basename, self.patient)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loading Baseline Mode from {}".format(model_path))
        else:
            print("Model is not exist in {}".format(model_path))
            exit()
        return

    @staticmethod
    def log_write(result, path='../log/log.txt'):
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(path):
            f = open(path, 'w')
        else:
            f = open(path, 'a')
        result_log = result + "\t" + time_stamp + '\n'
        print(result_log)
        f.write(result_log)
        f.close()
        print("Generating log!")
        return

    '''
    为了编码的方便，需要对最后的神经元进行处理，主要是其他的输出和SVM的不同的处理
    
    '''

    def cal_probability(self, output):
        if self.basename == 'cnnSvm':
            pred_y = [1 if d > 0 else 0 for d in output.data]
        else:
            pred_y = torch.max(output, 1)[1].data
            pred_y = pred_y.cpu()
        return pred_y

    @staticmethod
    def loss_svm(output, y):
        return torch.mean(torch.clamp(1 - output.t() * y, min=0))

    def train(self):
        mydata = MyData(path_train=self.train_path, path_test=self.test_path, path_val=self.val_path,
                        batch_size=self.batch_size,
                        few_shot=self.few_shot, few_shot_ratio=self.few_shot_ratio, isUnbalance=self.unbalance)

        train_data_loader = mydata.data_loader(None, mode='train')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.CrossEntropyLoss() if self.basename != 'cnnSvm' else Baselines.loss_svm

        acc_train, loss_train = [], []
        last_test_accuracy = 0
        with tqdm(total=self.epoch * len(train_data_loader)) as tq:

            for epoch in tqdm(range(self.epoch)):
                for step, (b_x, b_y, _, length, _) in enumerate(tqdm(train_data_loader)):  # gives batch data
                    b_x_g = b_x.cuda(self.gpu)

                    # 如果是SVM需要更改标签 将0/1 标签转化为-1/1的标签
                    if self.basename == 'cnnSvm':
                        b_y = torch.Tensor([1 if d == 1 else -1 for d in b_y])
                    b_y_g = b_y.cuda(self.gpu)
                    output = self.model(b_x_g)  #
                    loss = loss_func(output, b_y_g)  # cross entropy loss

                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    # SVM的计算方式不同需要更改计算的方式
                    pred_y = self.cal_probability(output)

                    res_tmp = [1 if pred_y[i] == b_y[i] else 0 for i in range(len(b_y))]
                    acc_train += res_tmp
                    loss_train.append(loss.data.cpu())
                    if step % 50 == 0:
                        acc_test, loss_test = [], []
                        for x_test, label_test, domain_test, length_test, _ in next(
                                mydata.next_batch_val_data(transform=None)):
                            # x_test = linear_matrix_normalization(x_test)
                            if self.gpu >= 0:
                                x_test, label_test, domain_test, length_test = x_test.cuda(self.gpu), label_test.cuda(
                                    self.gpu), domain_test.cuda(
                                    self.gpu), length_test.cuda(self.gpu)
                            with torch.no_grad():
                                label_output_test = self.model(x_test)
                                loss_label = loss_func(label_output_test, label_test)
                                pre_y_test = self.cal_probability(label_output_test)
                                acc_test += [1 if pre_y_test[i] == label_test[i] else 0 for i in range(len(label_test))]
                                loss_test.append(loss_label.data.cpu())
                        acc_train_avg = sum(acc_train) / len(acc_train)
                        loss_train_avg = sum(loss_train) / len(loss_train)

                        acc_test_avg = sum(acc_test) / len(acc_test)
                        loss_test_avg = sum(loss_test) / len(loss_test)

                        print(
                            'Epoch:{} | Step:{} | train loss:{:.6f} | val loss:{:.6f} | train accuracy:{:.5f} | val accuracy:{:.5f}'.format(
                                epoch, step, loss_train_avg, loss_test_avg, acc_train_avg, acc_test_avg))
                        acc_train.clear()
                        loss_train.clear()
                        if last_test_accuracy == 1:
                            last_test_accuracy = 0
                        if last_test_accuracy <= acc_test_avg:
                            self.save_mode()  # 保存较好的模型
                            print("Saving model...")
                            last_test_accuracy = acc_test_avg
                    tq.update(1)

    def evaluation(self, probability, y):
        '''
        评价指标的计算
        :param y:    实际的结果
        :return:  返回各个指标是的结果
        '''
        result = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1score': 0, 'auc': 0, 'far': 0}
        prey = [1 if x > 0.5 else 0 for x in probability]
        cal = IndicatorCalculation(prey, y)
        result['accuracy'] = cal.get_accuracy()
        result['precision'] = cal.get_precision()
        result['recall'] = cal.get_recall()
        result['f1score'] = cal.get_f1score()
        result['far'] = cal.get_far()
        result['auc'] = cal.get_auc(probability, y)

        return result

    def test(self):
        self.load_model()  # 加载模型
        mydata = MyData(path_train=self.train_path, path_test=self.test_path, path_val=self.val_path,
                        batch_size=self.batch_size)
        test_data_loader = mydata.data_loader(mode='test', transform=None)
        acc = []
        loss = []

        grand_true = []
        prediction = []
        probability = []

        loss_func = nn.CrossEntropyLoss() if self.basename != 'cnnSvm' else Baselines.loss_svm
        for step, (x, label, domain, length, ids) in enumerate(tqdm(test_data_loader)):
            if self.gpu >= 0:
                x, label, domain, length = x.cuda(self.gpu), label.cuda(self.gpu), domain.cuda(
                    self.gpu), length.cuda(
                    self.gpu)
            with torch.no_grad():
                label_output = self.model(x)
                loss_test = loss_func(label_output, label)
                loss_total = loss_test
                prey = self.cal_probability(label_output)

                y = label.cpu()
                acc += [1 if prey[i] == y[i] else 0 for i in range(len(y))]
                loss.append(loss_total.data.cpu())
                # if recoding:
                # ids_list += ids
                grand_true += [int(x) for x in y]
                prediction += [int(x) for x in prey]
                probability += [float(x) for x in
                                torch.softmax(label_output, dim=1)[:, 1]] if self.basename != 'cnnSvm' else [int(x) for
                                                                                                             x in prey]
        loss_avg = sum(loss) / len(loss)
        res = self.evaluation(probability, grand_true)

        result = "Baselins: {}|Patient {}|Data size:{}| test loss:{:.6f}| Accuracy:{:.5f} | Precision:" \
                 "{:.5f}| Recall:{:.5f}| F1score:{:.5f}| AUC:{:.5f} | FAR:{:.5f}".format(self.basename, self.patient,
                                                                                         len(acc),
                                                                                         loss_avg, res['accuracy'],
                                                                                         res['precision'],
                                                                                         res['recall'],
                                                                                         res['f1score'], res['auc'],
                                                                                         res['far'])
        self.log_write(result)
