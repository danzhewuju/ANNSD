import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Attention Is All You Need'''


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Transformer'
        # torch.cuda.set_device(gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.num_classes = 2  # 类别数
        self.num_epochs = 20  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.dim_model = 32
        self.hidden = 512
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2


class TransformerAttention(nn.Module):
    def __init__(self, gpu):
        super(TransformerAttention, self).__init__()
        self.config = Config()
        self.postion_embedding = Positional_Encoding(embed=32, pad_size=15, dropout=0.5, device=gpu)
        self.encoder = Encoder(dim_model=32, num_head=1, hidden=512, dropout=0.5)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(self.config.num_encoder)])

        self.fc1 = nn.Linear(480, 2)

    def forward(self, x):
        out = self.postion_embedding(x)  # embedding shape (128, 32, 300)
        for index, encoder in enumerate(self.encoders):
            if index == self.config.num_encoder - 1:
                out, attention = encoder(out, att=True)
            out = encoder(out)

        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out, attention


class Transformer(nn.Module):
    def __init__(self, gpu):
        super(Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(embed=32, pad_size=15, dropout=0.5, device=gpu)
        self.encoder = Encoder(dim_model=32, num_head=5, hidden=512, dropout=0.5)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(2)])

        self.fc1 = nn.Linear(480, 2)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):
        # x[0].shape = (32，15，32)
        # out = self.embedding(x[0])
        out = self.postion_embedding(x)  # embedding shape (128, 32, 300)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model=32, num_head=8, hidden=512, dropout=0.5):
        '''

        :param dim_model: 32 向量处理的长度
        :param num_head:  encoder 的部分
        :param hidden:    中间隐藏层的个数
        :param dropout:
        '''
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x, att=False):
        if att:
            out, attention = self.attention(x, att)
        else:
            out = self.attention(x, att)
        out = self.feed_forward(out)
        if att:
            return out, attention
        else:
            return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed=32, pad_size=15, dropout=0.5, device=0):
        '''

        :param embed:     字向量的维度，300
        :param pad_size:  每句话处理的长度
        :param dropout:   随机失活
        :param device:    显卡的位置
        '''
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None, att=False):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  #
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        # attention_npy = attention.cpu().data.numpy()
        # np.save('../log/attention.npy', attention_npy)
        context = torch.matmul(attention, V)
        if att:
            return context, attention
        else:
            return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()  # 计算attention
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, att):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  #
        #     mask = mask.repeat(self.num_head, 1, 1)
        scale = K.size(-1) ** -0.5  # 缩放因子
        if att:
            context, attention = self.attention(Q, K, V, scale, att)
        else:
            context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        if att:
            return out, attention
        else:
            return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
