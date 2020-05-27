import torch
from data_util import MyData
from model_util import DAN, ContrastiveLoss
import torch.nn as nn
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from util.util_file import trans_numpy_cv2, linear_matrix_normalization
import collections
import pandas as pd


class DanTrainer:
    def __init__(self, epoch=10, bath_size=16, lr=0.001, gpu=0, train_path=None, test_path=None, val_path=None,
                 model='train', encoder_name='vae', few_shot=True, few_show_ratio=0.2):
        self.epoch = epoch
        self.batch_size = bath_size
        self.lr = lr
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.encoder_name = encoder_name
        self.gpu = gpu
        self.few_shot = few_shot
        self.few_shot_ratio = few_show_ratio
        if gpu >= 0:
            self.model = DAN(gpu=gpu, model=model, encoder_name=encoder_name).cuda(gpu)  # 放入显存中
        else:
            self.model = DAN(gpu=gpu, model=model, encoder_name=encoder_name)  # 放入内存中

    def save_mode(self, save_path='../save_model'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_full_path = os.path.join(save_path, 'DAN_encoder_{}.pkl'.format(self.encoder_name))
        torch.save(self.model.state_dict(), save_full_path)
        print("Saving Model DAN in {}......".format(save_full_path))
        return

    def load_model(self, model_path='../save_model/DAN_encoder_{}.pkl'):
        model_path = model_path.format(self.encoder_name)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loading Mode DAN from {}".format(model_path))
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

        mydata = MyData(self.train_path, self.test_path, self.val_path, self.batch_size, few_shot=self.few_shot,
                        few_shot_ratio=self.few_shot_ratio)

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
        with tqdm(total=self.epoch * len(train_data_loader)) as pbar:
            for epoch in tqdm(range(self.epoch)):

                for step, (x, label, domain, length) in enumerate(tqdm(train_data_loader)):
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
                        loss_vae = 0
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
                        for x_test, label_test, domain_test, length_test in next(
                                mydata.next_batch_test_data(transform=None)):
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

                        print(
                            'Epoch:{} | Step:{} | train loss:{:.6f} | val loss:{:.6f} | train accuracy:{:.5f} | val accuracy:{:.5f}'.format(
                                epoch, step, loss_avg, test_loss_avg, accuracy_avg, test_accuracy_avg))
                        acc.clear()
                        loss.clear()
                        if last_test_accuracy == 1:
                            last_test_accuracy = 0
                        if last_test_accuracy <= test_accuracy_avg:
                            self.save_mode()  # 保存较好的模型
                            last_test_accuracy = test_accuracy_avg
                    pbar.update(1)

        if self.encoder_name == 'vae':
            info = {'loss_l': loss_prediction_vi, 'loss_d': loss_domain_discrimination_vi, 'loss_t': loss_vi,
                    'acc': acc_vi,
                    'loss_vae': loss_vae_vi,
                    'save_path': './draw/train_loss_{}.png'.format(self.encoder_name),
                    'model_info': "training information", 'show': False}
            self.draw_loss_plt(**info)
            info = {'loss_l': test_loss_prediction_vi, 'loss_d': test_loss_domain_discriminator_vi,
                    'loss_t': test_loss_vi,
                    'acc': test_acc_vi, 'loss_vae': test_loss_vae_vi,
                    'save_path': './draw/test_loss_{}.png'.format(self.encoder_name),
                    'model_info': "validation information", 'show': False}
            self.draw_loss_plt(**info)
        else:
            info = {'loss_l': loss_prediction_vi, 'loss_d': loss_domain_discrimination_vi, 'loss_t': loss_vi,
                    'acc': acc_vi,
                    'save_path': './draw/train_loss_{}.png'.format(self.encoder_name),
                    'model_info': "training information", 'show': False}
            self.draw_loss_plt(**info)
            info = {'loss_l': test_loss_prediction_vi, 'loss_d': test_loss_domain_discriminator_vi,
                    'loss_t': test_loss_vi,
                    'acc': test_acc_vi,
                    'save_path': './draw/test_loss_{}.png'.format(self.encoder_name),
                    'model_info': "validation information", 'show': False}
            self.draw_loss_plt(**info)

    def segment_statistic(self, prey, y, length):
        for i in range(len(prey)):
            if prey[i] == y[i]:
                self.result[int(length[i] // 500) + 1] += [1]
            else:
                self.result[int(length[i] // 500) + 1] += [0]

    def test(self):
        self.load_model()  # 加载模型
        mydata = MyData(self.train_path, self.test_path, self.val_path, self.batch_size)
        test_data_loader = mydata.data_loader(mode='test', transform=None)
        acc = []
        loss = []
        self.result = collections.defaultdict(list)
        loss_func = nn.CrossEntropyLoss()
        for step, (x, label, domain, length) in enumerate(tqdm(test_data_loader)):
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
                self.segment_statistic(prey, y, length.cpu())
        loss_avg = sum(loss) / len(loss)
        accuracy_avg = sum(acc) / len(acc)
        result = "Encoder:{}|Data size:{}| test loss:{:.6f}| Accuracy:{:.5f} ".format(self.encoder_name, len(acc),
                                                                                      loss_avg, accuracy_avg)
        self.log_write(result)
        # 分段统计信息表
        w, accs = [], []
        for l, p in self.result.items():
            acc = sum(p) / len(p)
            w.append(l)
            accs.append(acc)
        acc_data_frame = {'w': w, 'accs': accs}

        dataframe = pd.DataFrame(acc_data_frame)
        dataframe.to_csv('../log/segment_statistic.csv')
        print(dataframe)
