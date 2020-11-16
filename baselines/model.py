'''
增加一些baselines模型

1. CNNVoting
2. CNNTransformer
3. CNNLstm

'''

import sys

import torch
from torch import nn

sys.path.append('../')
import torch.nn.functional as F

from mymodel.Transformer_model import Transformer
from torch.nn.init import kaiming_normal_

# import sys
#
# sys.path.append('../')
# from util.util_tool import

'''
a new baseline : Recurrent Convolutional Neural Networks for Text Classification
paper link: http://zhengyima.com/my/pdfs/Textrcnn.pdf
'''


class clstm(nn.Module):
    def __init__(self, gpu=0, input_size=32, Resampling=500):
        super(clstm, self).__init__()
        self.input_size = input_size
        self.Resampling = Resampling
        self.gpu = gpu

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(6 * 31 * 32, 32)  # x_ y_ 和你输入的矩阵有关系

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=self.input_size,  # 输入向量的长度
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        # res = []
        # batch_size = x.size(0)
        # 需要对数据进行处理
        bat = x.shape[0]
        if self.gpu >= 0:
            res = torch.zeros((bat, 15, 32)).cuda(self.gpu)
        else:
            res = torch.zeros((bat, 15, 32))
        for i in range(bat):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.fc1(tmx)
                res[i][j] = tmx

        r_out, (h_n, h_c) = self.rnn(res, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


'''
a new baseline : An automated system for epilepsy detection using EEG brain signals based on deep learning approach
paper link: https://www.sciencedirect.com/science/article/pii/S0957417418302513
'''


class cnnVoting(nn.Module):

    def __init__(self, gpu=0, input_size=32, Resampling=500):
        super(cnnVoting, self).__init__()
        self.input_size = input_size
        self.Resampling = Resampling
        self.gpu = gpu

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(6 * 31 * 32, 32),  # x_ y_ 和你输入的矩阵有关系
            nn.Linear(32, 2)
        )

    '''
    需要对于数据进行处理;
    X=(batch_size, weight, height)
    '''

    def forward(self, x):
        batch_size = x.shape[0]
        if self.gpu >= 0:
            res = torch.zeros(batch_size, 2).cuda(self.gpu)
        else:
            res = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.layer5(tmx)
                tmx = tmx.reshape(-1)
                res[i] += tmx
            res[i] /= length
        return res


'''
a new baseline : Learning to Exploit Invariances in Clinical Time-Series Data using Sequence Transformer Networks
paper link: https://arxiv.org/abs/1808.06725

'''


class cnnTransformer(nn.Module):
    def __init__(self, gpu=0, input_size=32, Resampling=500):
        super(cnnTransformer, self).__init__()
        self.input_size = input_size
        self.Resampling = Resampling
        self.gpu = gpu  # 指定GPU
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(6 * 31 * 32, self.input_size)  # x_ y_ 和你输入的矩阵有关系,设定输出的维度的大小
        self.transformer = Transformer(self.gpu)

    def forward(self, x):
        batchSize = x.shape[0]
        if self.gpu >= 0:
            res = torch.zeros((batchSize, 15, 32)).cuda(self.gpu)
        else:
            res = torch.zeros((batchSize, 15, 32))
        for i in range(batchSize):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.fc1(tmx)
                res[i][j] = tmx
        out = self.transformer(res)
        return out


'''
a new baseline:Epileptic Seizure Prediction over EEG Data using Hybrid CNN-SVM Model with Edge Computing Services
paper link: https://www.matec-conferences.org/articles/matecconf/abs/2018/69/matecconf_cscc2018_03016/matecconf_cscc2018_03016.html

'''


class cnnSvm(nn.Module):
    def __init__(self, gpu=0, input_size=32, Resampling=500):
        super(cnnSvm, self).__init__()
        self.input_size = input_size  # 输入纬度的大小
        self.Resampling = Resampling
        self.gpu = gpu  # 指定GPU
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(6 * 31 * 32, 32),  # x_ y_ 和你输入的矩阵有关系
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        if self.gpu >= 0:
            res = torch.zeros(batch_size, 1).cuda(self.gpu)
        else:
            res = torch.zeros(batch_size, 1)
        for i in range(batch_size):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.layer5(tmx)
                tmx = tmx.reshape(-1)
                res[i] += tmx
            res[i] /= length
        return res


'''
Add a baseline： Very-deep-CNN(VDCNN)
paper link: https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1606.01781.pdf
github link : https://github.com/uvipen/Very-deep-cnn-pytorch/blob/master/train.py
'''


class ConvBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False,
                 downsampling=None):
        super(ConvBlock, self).__init__()

        self.downsampling = downsampling
        self.shortcut = shortcut
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, input):

        residual = input
        output = self.conv1(input)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)

        if self.shortcut:
            if self.downsampling is not None:
                residual = self.downsampling(input)
            output += residual

        output = self.relu2(output)

        return output


class VDCNN(nn.Module):

    def __init__(self, gpu, input_size, Resampling):
        super(VDCNN, self).__init__()
        n_classes = 2
        depth = 9
        n_fc_neurons = 2048
        shortcut = False
        embedding_dim = input_size
        self.gpu = gpu
        self.Resampling = Resampling

        layers = []
        fc_layers = []
        base_num_features = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(6 * 31 * 32, 32),  # x_ y_ 和你输入的矩阵有关系
        )

        layers.append(nn.Conv1d(embedding_dim, base_num_features, kernel_size=3, padding=1))

        if depth == 9:
            num_conv_block = [0, 0, 0, 0]
        elif depth == 17:
            num_conv_block = [1, 1, 1, 1]
        elif depth == 29:
            num_conv_block = [4, 4, 1, 1]
        elif depth == 49:
            num_conv_block = [7, 7, 4, 2]

        layers.append(ConvBlock(input_dim=base_num_features, n_filters=base_num_features, kernel_size=3, padding=1,
                                shortcut=shortcut))
        for _ in range(num_conv_block[0]):
            layers.append(ConvBlock(input_dim=base_num_features, n_filters=base_num_features, kernel_size=3, padding=1,
                                    shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(base_num_features, 2 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(2 * base_num_features))
        layers.append(
            ConvBlock(input_dim=base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[1]):
            layers.append(
                ConvBlock(input_dim=2 * base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(2 * base_num_features, 4 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(4 * base_num_features))
        layers.append(
            ConvBlock(input_dim=2 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[2]):
            layers.append(
                ConvBlock(input_dim=4 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(4 * base_num_features, 8 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(8 * base_num_features))
        layers.append(
            ConvBlock(input_dim=4 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[3]):
            layers.append(
                ConvBlock(input_dim=8 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))

        layers.append(nn.AdaptiveMaxPool1d(8))
        fc_layers.extend([nn.Linear(8 * 8 * base_num_features, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        batch_size = x.shape[0]
        if self.gpu >= 0:
            res = torch.zeros(batch_size, 15, 32).cuda(self.gpu)
        else:
            res = torch.zeros(batch_size, 15, 32)
        for i in range(batch_size):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.layer1(tmp_split)
                tmx = self.layer2(tmx)
                tmx = self.layer3(tmx)
                tmx = self.layer4(tmx)
                tmx = tmx.reshape(1, -1)  # 这里面的-1代表的是自适应的意思。
                tmx = self.layer5(tmx)
                tmx = tmx.reshape(-1)
                res[i][j] = tmx
        # output = self.embed(input)
        output = res.transpose(1, 2)
        output = self.layers(output)
        output = output.view(output.size(0), -1)
        output = self.fc_layers(output)

        return output


'''''
add a new baseline: Deep Pyramid Convolutional Neural Networks for Text Categorization
paper link: https://www.aclweb.org/anthology/P17-1052.pdf
github link :https://github.com/Cheneng/DPCNN 
'''''


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, gpu, input_size, Resampling):
        super(DPCNN, self).__init__()
        self.gpu = gpu
        self.Resampling = Resampling
        self.input_size = input_size

        # self.config = config
        self.channel_size = 15
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.fc = nn.Linear(6 * 31 * 32, 32)  # x_ y_ 和你输入的矩阵有关系

        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, input_size), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.channel_size, 2)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.gpu >= 0:
            res = torch.zeros(batch_size, 15, self.input_size).cuda(self.gpu)
        else:
            res = torch.zeros(batch_size, 15, self.input_size)
        for i in range(batch_size):
            tmp_x = x[i][0]
            length = tmp_x.shape[-1] // self.Resampling
            for j in range(length):
                tmp_split = tmp_x[:, self.Resampling * j:(j + 1) * self.Resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, 100, self.Resampling))
                tmx = self.embedding(tmp_split)
                tmx = tmx.reshape(-1)
                tmx = self.fc(tmx)
                tmx = tmx.reshape(-1)
                res[i][j] = tmx

        # Region embedding
        # x = self.embedding(res)
        # x = self.conv_region_embedding(x)  # [batch_size, channel_size, length, 1]
        # 修改此处的embedding
        res = torch.unsqueeze(res, len(res.shape))

        res = self.padding_conv(res)  # pad保证等长卷积，先通过激活函数再卷积
        res = self.act_fun(res)
        res = self.conv3(res)
        res = self.padding_conv(res)
        res = self.act_fun(res)
        res = self.conv3(res)

        while res.size()[-2] > 2:
            res = self._block(res)

        res = res.view(batch_size, 2 * self.channel_size)
        res = self.linear_out(res)

        return res

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    # def predict(self, x):
    #     self.eval()
    #     out = self.forward(x)
    #     predict_labels = torch.max(out, 1)[1]
    #     self.train(mode=True)
    #     return predict_labels
