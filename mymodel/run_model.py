import collections
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_util import MyData, SingleDataInfo, SingleDataset
from model_util import DAN, ContrastiveLoss
from util.seeg_utils import re_sampling, select_channel_data_mne, read_raw, read_annotations, read_edf_raw
from util.util_file import trans_numpy_cv2, IndicatorCalculation


class Dan:
    def __init__(self, patient, epoch, bath_size, lr, gpu, train_path, test_path, val_path, att_path, model,
                 encoder_name, few_shot, few_show_ratio, label_classifier_name, check_point, att, isUnbalance):
        self.patient = patient
        self.epoch = epoch
        self.batch_size = bath_size
        self.lr = lr
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.att_path = att_path
        self.encoder_name = encoder_name
        self.gpu = gpu
        self.few_shot = few_shot
        self.few_shot_ratio = few_show_ratio
        self.label_classifier_name = label_classifier_name
        self.check_point = check_point
        self.att = att  # 是否用于计算attention机制
        self.isUnbalance = isUnbalance

        if gpu >= 0:
            self.model = DAN(gpu=gpu, model=model, encoder_name=encoder_name,
                             label_classifier_name=label_classifier_name, att=att, isUnbalance=isUnbalance).cuda(
                gpu)  # 放入显存中
        else:
            self.model = DAN(gpu=gpu, model=model, encoder_name=encoder_name,
                             label_classifier_name=label_classifier_name, att=att, isUnbalance=isUnbalance)  # 放入内存中
        if self.check_point:
            self.load_model()  # 如果是断点训练
            print(" Start checkpoint training")

    def save_mode(self, save_path='../save_model'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_full_path = os.path.join(save_path,
                                      'DAN_encoder_{}_{}_{}.pkl'.format(self.encoder_name, self.label_classifier_name,
                                                                        self.patient))
        torch.save(self.model.state_dict(), save_full_path)
        print("Saving Model DAN in {}......".format(save_full_path))
        return

    def load_model(self, model_path='../save_model/DAN_encoder_{}_{}_{}.pkl'):

        model_path = model_path.format(self.encoder_name, self.label_classifier_name, self.patient)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(model_path)
            print("Loading Mode DAN from {}".format(self.patient))
        else:
            print("Model is not exist in {}".format(model_path))
        return

    def log_write(self, result, path='../log/log.txt'):
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

    def draw_loss_plt(self, *args, **kwargs):
        # 画出train或者test过程中的loss曲线
        loss_l, loss_d, loss_t, acc = kwargs['loss_l'], kwargs['loss_d'], kwargs['loss_t'], kwargs['acc']
        loss_vae = None
        if 'loss_vae' in kwargs.keys():
            loss_vae = kwargs['loss_vae']
        plot_save_path = kwargs['save_path']
        model_info = kwargs['model_info']
        show = kwargs['show']
        dir_name = os.path.dirname(plot_save_path)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Create dir {}".format(dir_name))
        plt.figure()
        plt.xlabel('Step')
        plt.ylabel('Loss/Accuracy')
        plt.title(model_info)
        x = range(len(acc))
        plt.plot(x, loss_l, label="Loss of label classifier")
        plt.plot(x, loss_d, label="Loss of domain discriminator")
        if 'loss_vae' in kwargs.keys():
            plt.plot(x, loss_vae, label='Loss of VAEs')
        plt.plot(x, loss_t, label='Total loss')
        plt.plot(x, acc, label='Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(plot_save_path)
        if show:
            plt.show()
        plt.close()
        print("The Picture has been saved.")

    def transform_data(self, x):
        '''
        模型数据的转化，需要添加高斯噪音
        :return:
        '''
        trans_data = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(96),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        x = trans_numpy_cv2(x)
        x = Image.fromarray(x)
        x = trans_data(x)
        result = np.array(x)
        result = result.reshape((result.shape[1:]))
        noise = 0.01 * np.random.rand(result.shape[0], result.shape[1])
        result += noise
        return result

    def train(self):  # 用于模型的训练

        mydata = MyData(self.train_path, self.test_path, self.val_path, self.att_path, self.batch_size,
                        few_shot=self.few_shot,
                        few_shot_ratio=self.few_shot_ratio, isUnbalance=self.isUnbalance)

        train_data_loader = mydata.data_loader(None, mode='train')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.CrossEntropyLoss()
        loss_func_domain = ContrastiveLoss()

        acc = []
        loss = []
        # 可视化的记录
        # 训练中的数据
        acc_vi = []
        loss_vi = []
        loss_prediction_vi = []
        loss_domain_discrimination_vi = []
        loss_vae_vi = []

        # 测试集中的数据
        test_acc_vi = []
        test_loss_vi = []
        test_loss_prediction_vi = []
        test_loss_domain_discriminator_vi = []
        test_loss_vae_vi = []

        last_test_accuracy = 0
        last_test_f1 = 0
        cal = IndicatorCalculation()
        with tqdm(total=self.epoch * len(train_data_loader)) as pbar:
            for epoch in tqdm(range(self.epoch)):

                for step, (x, label, domain, length, _) in enumerate(tqdm(train_data_loader)):
                    # x = linear_matrix_normalization(x)
                    if self.gpu >= 0:
                        x, label, domain, length = x.cuda(self.gpu), label.cuda(self.gpu), domain.cuda(
                            self.gpu), length.cuda(
                            self.gpu)
                    if self.encoder_name == 'vae':
                        label_output, domain_output_1, domain_output_2, domain_label, loss_vae = self.model(x, label,
                                                                                                            domain,
                                                                                                            length)
                    else:
                        label_output, domain_output_1, domain_output_2, domain_label = self.model(x, label, domain,
                                                                                                  length)
                    loss_label = loss_func(label_output, label)
                    loss_domain = loss_func_domain(domain_output_1, domain_output_2, domain_label)
                    if self.encoder_name == 'vae':
                        loss_total = (loss_label + loss_domain + loss_vae) / 3
                    else:
                        loss_total = 0.4 * loss_label + 0.6 * loss_domain

                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

                    pre_y = torch.max(label_output, 1)[1].data.cpu()
                    y = label.cpu()
                    acc += [1 if pre_y[i] == y[i] else 0 for i in range(len(y))]
                    loss.append(loss_total.data.cpu())
                    if step % 50 == 0:
                        # 数据可视化的整理
                        # 训练集的数据整理
                        loss_vi.append(loss_total.data.cpu())
                        loss_prediction_vi.append(loss_label.data.cpu())
                        loss_domain_discrimination_vi.append(loss_domain.data.cpu())
                        if self.encoder_name == 'vae': loss_vae_vi.append(loss_vae.data.cpu())

                        acc_test, test_loss = [], []
                        ground_test, prediction = [], []
                        for x_test, label_test, domain_test, length_test, _ in next(
                                mydata.next_batch_val_data(transform=None)):
                            # x_test = linear_matrix_normalization(x_test)
                            if self.gpu >= 0:
                                x_test, label_test, domain_test, length_test = x_test.cuda(self.gpu), label_test.cuda(
                                    self.gpu), domain_test.cuda(
                                    self.gpu), length_test.cuda(self.gpu)
                            with torch.no_grad():
                                if self.encoder_name == 'vae':
                                    label_output_test, domain_output_1, domain_output_2, domain_label, loss_vae = self.model(
                                        x_test, label_test, domain_test, length_test)
                                else:
                                    label_output_test, domain_output_1, domain_output_2, domain_label = self.model(
                                        x_test, label_test, domain_test, length_test)

                                loss_label = loss_func(label_output_test, label_test)
                                loss_domain = loss_func_domain(domain_output_1, domain_output_2, domain_label)
                                if self.encoder_name == 'vae':
                                    loss_total = (loss_label + loss_domain + loss_vae) / 3
                                else:
                                    loss_total = (loss_label + loss_domain) / 2

                                y_test = label_test.cpu()
                                pre_y_test = torch.max(label_output_test, 1)[1].data
                                acc_test += [1 if pre_y_test[i] == y_test[i] else 0 for i in range(len(y_test))]
                                test_loss.append(loss_total.data.cpu())
                                ground_test += y_test
                                prediction += pre_y_test
                        # 测试集的数据可视化的整理

                        test_loss_vi.append(loss_total.data.cpu())
                        test_loss_prediction_vi.append(loss_label.data.cpu())
                        test_loss_domain_discriminator_vi.append(loss_domain.data.cpu())
                        if self.encoder_name == 'vae': test_loss_vae_vi.append(loss_vae.data.cpu())

                        test_accuracy_avg = sum(acc_test) / len(acc_test)
                        test_loss_avg = sum(test_loss) / len(test_loss)
                        loss_avg = sum(loss) / len(loss)
                        accuracy_avg = sum(acc) / len(acc)

                        # 准确率的可视化
                        acc_vi.append(accuracy_avg)
                        test_acc_vi.append(test_accuracy_avg)

                        # 测试级效果
                        cal.set_values(prediction, ground_test)
                        f1_tmp = cal.get_f1score()

                        print(
                            'Epoch:{} | Step:{} | train loss:{:.6f} | val loss:{:.6f} | train accuracy:{:.5f} | val accuracy:{:.5f}| val f1score:{:.5f}'.format(
                                epoch, step, loss_avg, test_loss_avg, accuracy_avg, test_accuracy_avg, f1_tmp))
                        acc.clear()
                        loss.clear()
                        if last_test_f1 == 1:
                            last_test_f1 = 0
                        if last_test_f1 <= f1_tmp:
                            self.save_mode()  # 保存较好的模型
                            last_test_f1 = f1_tmp
                    pbar.update(1)

        # if self.encoder_name == 'vae':
        #     info = {'loss_l': loss_prediction_vi, 'loss_d': loss_domain_discrimination_vi, 'loss_t': loss_vi,
        #             'acc': acc_vi,
        #             'loss_vae': loss_vae_vi,
        #             'save_path': './draw/train_loss_{}.png'.format(self.encoder_name),
        #             'model_info': "training information", 'show': False}
        #     self.draw_loss_plt(**info)
        #     info = {'loss_l': test_loss_prediction_vi, 'loss_d': test_loss_domain_discriminator_vi,
        #             'loss_t': test_loss_vi,
        #             'acc': test_acc_vi, 'loss_vae': test_loss_vae_vi,
        #             'save_path': './draw/test_loss_{}.png'.format(self.encoder_name),
        #             'model_info': "validation information", 'show': False}
        #     self.draw_loss_plt(**info)
        # else:
        #     info = {'loss_l': loss_prediction_vi, 'loss_d': loss_domain_discrimination_vi, 'loss_t': loss_vi,
        #             'acc': acc_vi,
        #             'save_path': './draw/train_loss_{}.png'.format(self.encoder_name),
        #             'model_info': "training information", 'show': False}
        #     self.draw_loss_plt(**info)
        #     info = {'loss_l': test_loss_prediction_vi, 'loss_d': test_loss_domain_discriminator_vi,
        #             'loss_t': test_loss_vi,
        #             'acc': test_acc_vi,
        #             'save_path': './draw/test_loss_{}.png'.format(self.encoder_name),
        #             'model_info': "validation information", 'show': False}
        #     self.draw_loss_plt(**info)

    def segment_statistic(self, prey, y, length):
        '''
        模型分段预测的情况,针对不同长长度的分段数据
        :param prey: 预测的标签
        :param y: 实际的标签
        :param length: 输入的长度
        :return:
        '''
        for i in range(len(prey)):
            if prey[i] == y[i]:
                self.result[int(length[i] // 500)] += [1]
            else:
                self.result[int(length[i] // 500)] += [0]

    def save_segment_statistic_info(self):
        '''
        模型预测分段信息进行保存
        :return:
        '''
        w, accs, vars = [], [], []
        # 增加方差的计算
        epoch = 5  # 将数据分组用于计算方差

        for l, p in self.result.items():
            acc_ep = []
            batch_size = len(p) // epoch  # 每一个batch size 的大小
            for j in range(epoch):
                tmp = p[j * batch_size:(j + 1) * batch_size]
                acc_ep.append(sum(tmp) / len(tmp))
            acc = np.mean(acc_ep)
            var = np.std(acc_ep)
            w.append(l)
            accs.append(acc)
            vars.append(var)
        acc_data_frame = {'w': w, 'accs': accs, 'var': vars}

        dataframe = pd.DataFrame(acc_data_frame)
        dataframe.to_csv('../log/segment_statistic_{}_{}.csv'.format(self.encoder_name, self.label_classifier_name))
        print(dataframe)

    def save_all_input_prediction_result(self, ids_list, grand_true, prediction,
                                         save_path='../log/prediction_result.csv'):
        """

        :return: saving all files' prediction result
        """
        data = {'id': ids_list, 'grand true': grand_true, 'prediction': prediction}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(save_path)
        print('Saving success!')
        return

    def evaluation(self, probability, y):
        '''
        评价指标的计算
        :param probability: 预测的结果
        :param y:    实际的结果
        :return:  返回各个指标是的结果
        '''
        result = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1score': 0, 'auc': 0}
        prey = [1 if x > 0.5 else 0 for x in probability]
        cal = IndicatorCalculation(prey, y)
        result['accuracy'] = cal.get_accuracy()
        result['precision'] = cal.get_precision()
        result['recall'] = cal.get_recall()
        result['f1score'] = cal.get_f1score()
        result['auc'] = cal.get_auc(probability, y)
        result['far'] = cal.get_far()
        return result

    def test(self, recoding=False):
        '''

        :param recoding: 是否将每一个样本的预测结果记录下来
        :return:
        '''
        self.load_model()  # 加载模型
        mydata = MyData(self.train_path, self.test_path, self.val_path, self.att_path, self.batch_size)
        test_data_loader = mydata.data_loader(mode='test', transform=None)
        acc = []
        loss = []

        ids_list = []
        grand_true = []
        prediction = []
        probability = []

        self.result = collections.defaultdict(list)
        loss_func = nn.CrossEntropyLoss()
        for step, (x, label, domain, length, ids) in enumerate(tqdm(test_data_loader)):
            if self.gpu >= 0:
                x, label, domain, length = x.cuda(self.gpu), label.cuda(self.gpu), domain.cuda(
                    self.gpu), length.cuda(
                    self.gpu)
            with torch.no_grad():
                label_output = self.model(x, label, domain, length)
                loss_label = loss_func(label_output, label)
                loss_total = loss_label
                prey = torch.max(label_output, 1)[1].data.cpu()

                y = label.cpu()
                acc += [1 if prey[i] == y[i] else 0 for i in range(len(y))]
                loss.append(loss_total.data.cpu())
                # if recoding:
                ids_list += ids
                grand_true += [int(x) for x in y]
                prediction += [int(x) for x in prey]
                probability += [float(x) for x in torch.softmax(label_output, dim=1)[:, 1]]
                self.segment_statistic(prey, y, length.cpu())
        loss_avg = sum(loss) / len(loss)
        res = self.evaluation(probability, grand_true)

        message = "Unbalance" if self.isUnbalance > 1 else "balance"

        result = "{} |Encoder:{}|Label classifier {}|Patient {}|Data size:{}| test loss:{:.6f}| Accuracy:{:.5f} | Precision:" \
                 "{:.5f}| Recall:{:.5f}| F1score:{:.5f}| AUC:{:.5f}| FAR:{:.5f}".format(message,
                                                                                        self.encoder_name,
                                                                                        self.label_classifier_name,
                                                                                        self.patient, len(acc),
                                                                                        loss_avg, res['accuracy'],
                                                                                        res['precision'], res['recall'],
                                                                                        res['f1score'], res['auc'],
                                                                                        res['far'])
        self.log_write(result)
        if recoding:  # 如果开启了记录模式，模型会记录所有的文件的预测结果
            self.save_all_input_prediction_result(ids_list, grand_true, prediction)

    def prediction_batch_real_data(self, file_path_list, label, data_length, config_path=None):
        """
        方法的功能是对批量的数据进行检测

        :param file_path_list: 文件的路径列表
        :param label:  所属的标签
        :param data_length: 数据的窗口的长度
        :param config_path: 信道等信息的存储位置
        :return: None
        """
        # 需要设定最低的时间，本次实验设定的还是60s
        MIN_DURATION = 60
        data_info = pd.read_csv(file_path_list)
        for index, row in data_info.iterrows():
            file_path = row['Path']
            pre_seizure_duration = row['Pre_Seizure Duration(s)']
            ACC = 0
            if pre_seizure_duration >= MIN_DURATION:
                # 这里必须要关闭日志记录，否则会和手动保存的日志冲突
                ACC = self.prediction_real_data(file_path, label, None, data_length, config_path=config_path,
                                                log_flag=False)
            data_info[index, 'Accuracy'] = ACC
        print("All test finished!")
        return

    def prediction_real_data(self, file_path, label, save_file, data_length, config_path=None, log_flag=True):
        """
        :function 在实际的情况下的模型的准确率,单个文件的预测结果
        :param file_path: 原始数据的文件路径
        :param label: 文件的实际标签
        :param save_file: 保存结果的文件路径
        :param data_length: 设定的数据划分长度
        :param config_path: 相关的配置文件的目录
        :return:
        """

        def read_data(path):
            """
            本模块是专门用来读取癫痫发作前睡眠的数据
            :param path: 文件路径
            :return:
            """
            MAX_LENGTH = 15
            fix = path.split('.')[-1]  # 后缀
            if fix == "npy":
                data = np.load(path)
            elif fix == "fif":
                data = read_raw(path)
            elif fix == "edf":
                # 这里对应着原始文件，需要选取特定范围的数据，流程上希望自动完成对数据的截取操作
                data = read_edf_raw(path)
                annotation = read_annotations(data)
                onset = annotation['onset']
                if len(onset) > 3:
                    # 读取onset的时间
                    end_time = int(onset[3]) - MAX_LENGTH
                    # 如果预留的时间小于30s无法执行 此时要留一个30s的gap;
                    if end_time <= 0:
                        data = None
                        print("长度不足{}s".format(MAX_LENGTH))
                    else:
                        data = data.crop(0, end_time)
                else:
                    data = None
            else:
                pass
            return data

        self.load_model()  # 加载模型 加载模型, 这里需要直接手动的指定模型
        resampling = 500
        label_dict = {'pre_seizure': 1, 'non_seizure': 0}
        """
        暂时不需要该模块，如果需要对信道进行重新排序可能需要相关模块
        """

        data = read_data(file_path)
        # read_raw(file_path)
        # 数据读取失败
        if data == None:
            assert "data loading filed!"
            print("data loading filed!")
            exit()

        data = re_sampling(data, fz=500)  # 对于数据进行重采样
        # -------------------------------------------------------------------------------------------------------------

        # 研究信道排序对于模型的影响，去掉该模块基本没影响， 最新的实验表面该模块对于模型依然有较大的影响
        with open(config_path, 'r') as f:
            config = json.load(f)
        key = "{}_data_path".format(self.patient)
        channel_path = config[key]["data_channel_path"]  # 获取相关的保存位置信息
        channel_name = pd.read_csv(channel_path)
        channels_name = list(channel_name['chan_name'])
        data = select_channel_data_mne(data, channels_name)

        # -------------------------------------------------------------------------------------------------------------

        new_data, _ = data[:, :]  # 获取原始数据的形式
        single_data_info = SingleDataInfo(new_data, label, data_length=data_length)
        # 由单个文件构成的数据集
        single_dataset = SingleDataset(single_data_info.input, single_data_info.time_info, single_data_info.label)
        dataloader = DataLoader(single_dataset, batch_size=self.batch_size, shuffle=False)

        ids_list, prediction, probability = [], [], []
        label_int = label_dict[label]
        loss_func = nn.CrossEntropyLoss()
        for step, (x, y, time_) in enumerate(tqdm(dataloader)):
            if self.gpu >= 0:
                x, y = x.cuda(self.gpu), y.cuda(self.gpu)
            with torch.no_grad():
                inputLength = [data_length * resampling] * len(x)
                label_output = self.model(x, y, None, inputLength)
                prey = torch.max(label_output, 1)[1].data.cpu()

                ids_list += ["{}_{}_{}_{}".format(os.path.basename(file_path), self.patient, label, t // resampling) for
                             t in time_[0]]
                prediction += [int(x) for x in prey]
        probability += [1 if t == label_int else 0 for t in prediction]
        accuracy = sum(probability) / len(probability)
        if log_flag:
            log = "Encoder:{}|Label classifier {}|Patient {}|Data size:{}|Real data|Accuracy:{:.5f}".format(
                self.encoder_name, self.label_classifier_name, self.patient, len(probability), accuracy)
            self.log_write(log)
            # 文件日志的写入
            result = {'id': ids_list, 'ground truth': [label_int] * len(prediction), 'prediction': prediction}
            dataframe = pd.DataFrame(result)
            header = True if not os.path.exists(save_file) else False  # 判断文件在不在
            dataframe.to_csv(save_file, index=False, mode='a', header=header)

            print("All information has been save in {}".format(save_file))
        return accuracy

    def save_attention_matrix(self, attention_matrix, ids, result, save_dir='../log/attention'):
        '''

        :param attention_matrix: 需要写入的attention矩阵 shape(batch_size, maxlength, maxlength)
        :param ids:  每一个原始文件对应的id序号
        :param result : 预测的结果
        :param save_dir: 保存文件夹

        :return:
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Dir {} was created".format(save_dir))
        for i in range(attention_matrix.shape[0]):
            if result[i] == 1:  # 只有预测正确的才会保存attention
                data = attention_matrix[i]
                name = ids[i]
                save_path = os.path.join(save_dir, name + '.npy')
                np.save(save_path, data)
        print("All files has been saved!")
        return True

    def test_attention(self):
        self.load_model()  # 加载模型
        mydata = MyData(self.train_path, self.test_path, self.val_path, self.att_path, self.batch_size)
        test_data_loader = mydata.data_loader(mode='attention', transform=None)
        acc = []
        loss = []
        loss_func = nn.CrossEntropyLoss()
        for step, (x, label, domain, length, ids) in enumerate(tqdm(test_data_loader)):
            if self.gpu >= 0:
                x, label, domain, length = x.cuda(self.gpu), label.cuda(self.gpu), domain.cuda(
                    self.gpu), length.cuda(
                    self.gpu)
            with torch.no_grad():
                label_output, attention = self.model(x, label, domain, length)
                loss_label = loss_func(label_output, label)
                loss_total = loss_label
                prey = torch.max(label_output, 1)[1].data.cpu()
                y = label.cpu()
                tmp = [1 if prey[i] == y[i] else 0 for i in range(len(y))]
                acc += tmp
                loss.append(loss_total.data.cpu())
                self.save_attention_matrix(attention.cpu().data.numpy(), ids, tmp)
        loss_avg = sum(loss) / len(loss)
        accuracy_avg = sum(acc) / len(acc)
        result = "Encoder:{}|Data size:{}| test loss:{:.6f}| Accuracy:{:.5f} ".format(self.encoder_name, len(acc),
                                                                                      loss_avg, accuracy_avg)
        print(result)
