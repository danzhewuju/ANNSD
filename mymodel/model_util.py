import torch.nn as nn
from torch.autograd import Function
import torch


class DAN(nn.Module):
    def __init__(self, input_shape=(100, 500), c_dim=32, gpu=0, resampling=500, model='train'):
        '''

        :param c_dim:  表示的是encoder的输出,同时也是预测网络和对抗网络的输出
        '''
        # 需要构建网络部分 1.编码器， 2.标签， 3.对抗样本判别
        super(DAN, self).__init__()
        self.model = model
        self.input_shape = input_shape
        self.gpu = gpu  # 是否指定gpu
        self.resampling = resampling
        self.dim = c_dim

        # encoder 的数据部分 输入的格式是：
        self.encoder = nn.Sequential(
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
        self.fc1 = nn.Linear(6 * 31 * 32, c_dim)  # x_ y_ 和你输入的矩阵有关系

        self.label_classifier = nn.LSTM(
            input_size=c_dim,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.label_fc = nn.Linear(64, 2)

        self.domain_classifier = nn.LSTM(
            input_size=c_dim,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.domain_fc = nn.Linear(64, 5)

    def trans_data(self, x, length):
        '''
        转换数据的格式，对于数据按照要求进行编码
        :param x:
        :return:
        '''
        max_length = length[0] // self.resampling
        bat = x.shape[0]
        if self.gpu >=0 :
            res = torch.zeros((bat, max_length, self.dim)).cuda(self.gpu)
        else:
            res = torch.zeros((bat, max_length, self.dim))
        for i in range(bat):
            tmp_x = x[i][0]
            l = length[i] // self.resampling
            for j in range(l - 1):
                tmp_split = tmp_x[:, self.resampling * j:(j + 1) * self.resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, self.input_shape[0], self.input_shape[1]))
                tmx = self.encoder(tmp_split)
                tmx = self.fc1(tmx.reshape(1, -1))
                res[i][j] = tmx
        return res

    def forward(self, x, label, domain, length, alpha=1):
        code = self.trans_data(x, length)
        label_tmp, (_, _) = self.label_classifier(code)
        y_label = self.label_fc(label_tmp[:, -1, :])
        if self.model == 'train':
            reverse_feature = ReverseLayerF.apply(code, alpha)  # 对抗样本需要经过GRL模块
            domain_tmp,(_, _) = self.domain_classifier(reverse_feature)
            y_domain = self.domain_fc(domain_tmp[:, -1, :])
            return y_label, y_domain
        else:
            return y_label


class ReverseLayerF(Function):  # GRL模块，GRL模块在反向传播的过程中进行了梯度的反转
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
