import torch
from data_util import MyData
from model_util import DAN
import torch.nn as nn
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt


class DanTrainer:
    def __init__(self, epoch=10, bath_size=16, lr=0.001, GPU=0, train_path=None, test_path=None, val_path=None,
                 model='train'):
        self.epoch = epoch
        self.batch_size = bath_size
        self.lr = lr
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.gpu = GPU
        if GPU is not None:
            self.model = DAN(gpu=GPU, model=model).cuda(GPU)  # 放入显存中
        else:
            self.model = DAN(gpu=GPU, model=model)  # 放入内存中

    def save_mode(self, save_path='../save_model'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_full_path = os.path.join(save_path, 'DAN.pkl')
        torch.save(self.model.state_dict(), save_full_path)
        print("Saving Model DAN in {}......".format(save_full_path))
        return

    def load_model(self, model_path='../save_model/DAN.pkl'):
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

    def draw_loss_plt(*args, **kwargs):
        # 画出train或者test过程中的loss曲线
        loss_l, loss_d, loss_t, acc = kwargs['loss_l'], kwargs['loss_d'], kwargs['loss_t'], kwargs['acc']
        plot_save_path = kwargs['save_path']
        model_info = kwargs['model_info']
        dir_name = os.path.dirname(plot_save_path)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Create dir {}".format(dir_name))
        plt.plot()
        plt.xlabel('Step')
        plt.ylabel('Loss/Accuracy')
        plt.title(model_info)
        x = range(len(acc))
        plt.plot(x, loss_l, label="Loss of label classifier")
        plt.plot(x, loss_d, label="Loss of domain discriminator")
        plt.plot(x, loss_t, label='Total loss')
        plt.plot(x, acc, label='Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(plot_save_path)
        plt.show()
        print("The Picture has been saved.")

    def train(self):  # 用于模型的训练
        mydata = MyData(self.train_path, self.test_path, self.val_path, self.batch_size)

        train_data_loader = mydata.data_loader(mode='train')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.CrossEntropyLoss()

        acc = []
        loss = []
        # 可视化的记录
        # 训练中的数据
        acc_vi = []
        loss_vi = []
        loss_prediction_vi = []
        loss_domain_discrimination_vi = []

        # 测试集中的数据
        test_acc_vi = []
        test_loss_vi = []
        test_loss_prediction_vi = []
        test_loss_domain_discriminator_vi = []

        last_test_accuracy = 0

        for epoch in tqdm(range(self.epoch)):

            for step, (x, label, domain, length) in tqdm(enumerate(train_data_loader)):
                if self.gpu is not None:
                    x, label, domain, length = x.cuda(self.gpu), label.cuda(self.gpu), domain.cuda(
                        self.gpu), length.cuda(
                        self.gpu)
                label_output, domain_output = self.model(x, label, domain, length)
                loss_label = loss_func(label_output, label)
                loss_domain = loss_func(domain_output, domain)
                loss_total = (loss_label + loss_domain) * 0.5
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                pre_y = torch.max(label_output, 1)[1].data.cpu()
                y = label.cpu()
                acc += [1 if pre_y[i] == y[i] else 0 for i in range(len(y))]
                loss.append(loss_total.data.cpu())
                loss_vi.append(loss_total.data.cpu())
                if step % 50 == 0:
                    # 数据可视化的整理
                    # 训练集的数据整理
                    loss_vi.append(loss_total.data.cpu())
                    loss_prediction_vi.append(loss_label.data.cpu())
                    loss_domain_discrimination_vi.append(loss_domain.data.cpu())

                    acc_test, test_loss = [], []
                    for x_test, label_test, domain_test, length_test in next(mydata.next_batch_test_data()):
                        if self.gpu is not None:
                            x_test, label_test, domain_test, length_test = x_test.cuda(self.gpu), label_test.cuda(
                                self.gpu), domain_test.cuda(
                                self.gpu), length_test.cuda(self.gpu)
                        with torch.no_grad():
                            label_output_test, domain_output_test = self.model(x_test, label_test, domain_test,
                                                                               length_test)
                            loss_label = loss_func(label_output_test, label_test)
                            loss_domain = loss_func(domain_output_test, domain_test)
                            loss_total = (loss_label + loss_domain) * 0.5

                            y_test = label_test.cpu()
                            pre_y_test = torch.max(label_output_test, 1)[1].data
                            acc_test += [1 if pre_y_test[i] == y_test[i] else 0 for i in range(len(y_test))]
                            test_loss.append(loss_total.data.cpu())
                    # 测试集的数据可视化的整理

                    test_loss_vi.append(loss_total.data.cpu())
                    test_loss_prediction_vi.append(loss_label.data.cpu())
                    test_loss_domain_discriminator_vi.append(loss_domain.data.cpu())

                    test_accuracy_avg = sum(acc_test) / len(acc_test)
                    test_loss_avg = sum(test_loss) / len(test_loss)
                    loss_avg = sum(loss) / len(loss)
                    accuracy_avg = sum(acc) / len(acc)

                    # 准确率的可视化
                    acc_vi.append(accuracy_avg)
                    test_acc_vi.append(test_accuracy_avg)

                    print(
                        'Epoch:{} | Step:{} | train loss:{:.6f} | test loss:{:.6f} | train accuracy:{:.5f} | test accuracy:{:.5f} |'.format(
                            epoch, step, loss_avg, test_loss_avg, accuracy_avg, test_accuracy_avg))
                    acc.clear()
                    loss.clear()
                    if last_test_accuracy == 1:
                        last_test_accuracy = 0
                    if last_test_accuracy <= test_accuracy_avg:
                        self.save_mode()  # 保存较好的模型
                        last_test_accuracy = test_accuracy_avg
        info = {'loss_l': loss_prediction_vi, 'loss_d': loss_domain_discrimination_vi, 'loss_t': loss_vi, 'acc': acc_vi,
                'save_path': './draw/train_loss.png',
                'model_info': "training information"}
        self.draw_loss_plt(**info)
        info = {'loss_l': test_loss_vi, 'loss_d': test_loss_domain_discriminator_vi, 'loss_t': test_loss_vi, 'acc': test_acc_vi,
                'save_path': './draw/test_loss.png',
                'model_info': "test information"}
        self.draw_loss_plt(**info)

    def val(self):
        self.load_model()  # 加载模型
        mydata = MyData(self.train_path, self.test_path, self.val_path, self.batch_size)
        val_data_loader = mydata.data_loader(mode='val')
        acc = []
        loss = []
        loss_func = nn.CrossEntropyLoss()
        for step, (x, label, domain, length) in tqdm(enumerate(val_data_loader)):
            if self.gpu is not None:
                x, label, domain, length = x.cuda(self.gpu), label.cuda(self.gpu), domain.cuda(
                    self.gpu), length.cuda(
                    self.gpu)
            with torch.no_grad():
                label_output = self.model(x, label, domain, length)
                loss_label = loss_func(label_output, label)
                loss_total = loss_label
                pre_y = torch.max(label_output, 1)[1].data.cpu()
                y = label.cpu()
                acc += [1 if pre_y[i] == y[i] else 0 for i in range(len(y))]
                loss.append(loss_total.data.cpu())
        loss_avg = sum(loss) / len(loss)
        accuracy_avg = sum(acc) / len(acc)
        result = "Data size:{}| Val loss:{:.6f}| Accuracy:{:.5f} ".format(len(acc), loss_avg, accuracy_avg)
        self.log_write(result)
