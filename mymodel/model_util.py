import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import random
from Transformer import Transformer, TransformerAttention
from util.util_file import linear_matrix_normalization
import sys

sys.path.append('../')
from pytorch_pretrained import BertModel, BertTokenizer


class VAE(nn.Module):
    def __init__(self, input_shape=(100, 500), c_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, c_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(c_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_shape[0] * input_shape[1]),
            nn.Sigmoid()  # compress to a range (0, 1)
        )
        self.input_shape = input_shape

    def reparameterize(self, mu, logval):
        std = torch.exp(0.5 * logval)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_linear = torch.reshape(x, (-1, self.input_shape[0] * self.input_shape[1]))
        z = self.encoder(x_linear)
        # mu = self.fc1(coded)
        # logvar = self.fc2(coded)
        # z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        decoded = torch.reshape(decoded, (self.input_shape[0], self.input_shape[1]))
        return z, decoded


class CNNEncoder(nn.Module):
    def __init__(self, input_shape=(100, 500), c_dim=32):
        super(CNNEncoder, self).__init__()
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
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(6 * 31 * 32, c_dim)  # x_ y_ 和你输入的矩阵有关系

    def forward(self, x):
        x = self.encoder(x)
        x = torch.reshape(x, (1, -1))
        x = self.fc1(x)
        return x


class DAN(nn.Module):
    def __init__(self, input_shape=(100, 500), c_dim=32, gpu=0, resampling=500, model='train', encoder_name='vae',
                 label_classifier_name='lstm'):
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
        self.encoder_name = encoder_name
        self.label_classifier_name = label_classifier_name
        if encoder_name == 'vae':  # 使用vae作为编码器
            self.encoder = VAE(input_shape, c_dim=c_dim)  # 注意此时的训练方式，是将前一个的编码方式和我下一个的编码方式进行交叉求和

        else:
            self.encoder = CNNEncoder(input_shape, c_dim=c_dim)

        if label_classifier_name == 'lstm':
            self.label_classifier = nn.LSTM(
                input_size=c_dim,
                hidden_size=64,  # rnn hidden unit
                num_layers=1,  # number of rnn layer
                batch_first=True,
                bidirectional=True
                # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
            self.tanh1 = nn.Tanh()
            self.w = nn.Parameter(torch.zeros(64 * 2))
            self.tanh2 = nn.Tanh()
            self.fc1 = nn.Linear(64 * 2, 64)
            self.label_fc = nn.Linear(64, 2)
        else:  # 使用transformer 模型
            self.label_classifier = Transformer()
            # self.attention_model = TransformerAttention()
            # self.label_fc = nn.Linear(768, 2)

        self.domain_classifier = nn.LSTM(
            input_size=c_dim,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.domain_fc = nn.Linear(64, 2)  # 判断病人是否来自于同一个人

    def trans_data(self, x, length):
        '''
        转换数据的格式，对于数据按照要求进行编码
        :param x:
        :return:
        '''
        if self.label_classifier_name == 'transformer':
            max_length = 15
        else:
            max_length = length[0] // self.resampling
        bat = x.shape[0]
        if self.encoder_name == 'vae':
            loss_func = nn.functional.mse_loss
            loss_vae = torch.tensor(0.0, requires_grad=True).cuda(self.gpu)
            count = 0

        if self.gpu >= 0:
            res = torch.zeros((bat, max_length, self.dim)).cuda(self.gpu)
        else:
            res = torch.zeros((bat, max_length, self.dim))
        for i in range(bat):
            tmp_x = x[i][0]
            l = length[i] // self.resampling
            for j in range(l):
                tmp_split = tmp_x[:, self.resampling * j:(j + 1) * self.resampling]
                tmp_split = torch.reshape(tmp_split, (1, 1, self.input_shape[0], self.input_shape[1]))
                if self.encoder_name == 'vae':
                    z, decode = self.encoder(tmp_split)
                    if j < l - 1:
                        next_x = tmp_x[:, self.resampling * (j + 1):(j + 2) * self.resampling]

                        loss_vae += loss_func(decode, next_x)
                        count += 1
                else:
                    z = self.encoder(tmp_split)
                res[i][j] = z
        if self.encoder_name == 'vae':
            return res, loss_vae / count
        else:
            return res

    def forward(self, x, label, domain, length, alpha=1):
        if self.encoder_name == 'vae':
            code_x1, loss_vae = self.trans_data(x, length)
        else:
            code_x1 = self.trans_data(x, length)
        if self.label_classifier_name == 'lstm':  # 用不同的模型进行判别
            label_tmp, _ = self.label_classifier(code_x1)  # 使用lstm进行判别
            # 引入attention机制
            M = self.tanh1(label_tmp)
            alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
            out = label_tmp * alpha
            out = torch.sum(out, 1)
            out = F.relu(out)
            out = self.fc1(out)
            y_label = self.label_fc(out)
        else:
            y_label = self.label_classifier(code_x1)  # 利用transformer模型
        if self.model == 'train':  # 模型的成对训练
            if self.gpu >= 0:
                code_x2 = torch.zeros(code_x1.shape).cuda(self.gpu)
                domain_label = torch.zeros(code_x1.shape[0]).cuda(self.gpu)
            else:
                code_x2 = torch.zeros(code_x1.shape)
                domain_label = torch.zeros(code_x1.shape[0])
            code_x2_order = list(range(domain.shape[0]))
            pre_half, behind_half = code_x2_order[:len(code_x2_order) // 2], code_x2_order[len(code_x2_order) // 2:]
            random.shuffle(behind_half)  # 只打乱一半的数据, 否则的数据均衡会严重失调, 不进行打乱的情况会导致数据不均衡
            code_x2_order = pre_half + behind_half
            # random.shuffle(code_x2_order)  # 打乱全部数据
            for i, p in enumerate(code_x2_order):
                code_x2[i] = code_x1[p]
                if domain[i] == domain[p]:
                    domain_label[i] = 1
                else:
                    domain_label[i] = 0

            reverse_feature_x1 = ReverseLayerF.apply(code_x1, alpha)  # 对抗样本需要经过GRL模块
            reverse_feature_x2 = ReverseLayerF.apply(code_x2, alpha)  # 对抗样本需要经过GRL模块
            domain_tmp_x1, (_, _) = self.domain_classifier(reverse_feature_x1)
            domain_tmp_x2, (_, _) = self.domain_classifier(reverse_feature_x2)
            y_domain_1 = self.domain_fc(domain_tmp_x1[:, -1, :])
            y_domain_2 = self.domain_fc(domain_tmp_x2[:, -1, :])
            if self.encoder_name == 'vae':
                return y_label, y_domain_1, y_domain_2, domain_label, loss_vae  # 如果使用的是vae的编码，需要将vae的损失函数传递出去
            else:
                return y_label, y_domain_1, y_domain_2, domain_label


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


# 定义对抗的损失函数
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, use_domain=1, size_average=True):
        distances = 1e3 * (output2 - output1).pow(2).sum(1) + 0.001  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        losses = losses * use_domain
        return losses.mean() if size_average else losses.sum()
