"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
torchvision
"""
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from util.util_tool import *
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8
TIME_STEP = 100  # rnn time step / image height
INPUT_SIZE = 1000  # rnn input size / image width
LR = 0.001  # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data

TRAIN_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/train.csv"
TEST_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG_Timing/preprocess/test.csv"

data_train = Data_info(TRAIN_PATH)
data_test = Data_info(TEST_PATH)
train_data = MyDataset(data_train.data)  # 作为训练集
test_data = MyDataset(data_test.data)  # 作为测试集
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


# def train_transform(x):
#     trans_data = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         # transforms.RandomCrop(96),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     x = trans_numpy_cv2(x)
#     x = Image.fromarray(x)
#     x = trans_data(x)
#     result = np.array(x)
#     result = result.reshape((result.shape[1:]))
#     noise = np.random.rand(result.shape[0], result.shape[1])
#     result += noise
#     return result

# Mnist digital dataset
# train_data = dsets.MNIST(
#     root='./mnist/',
#     train=True,  # this is training data
#     transform=transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
#     # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#     download=DOWNLOAD_MNIST,  # download it if you don't have it
# )

# plot one example
# print(train_data.train_data.size())  # (60000, 28, 28)
# print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
# test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255.  # shape (2000, 28, 28) value in range(0,1)
# test_y = test_data.test_labels.numpy()[:2000]  # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        # b_x = b_x
        # b_y = b_y
        b_x = b_x.view(-1, 100, 1000)  # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = rnn(b_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            result = [1 if x == y else 0 for x, y in zip(pred_y, b_y)]
            accuracy = sum(result) / len(result)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

torch.save(rnn.state_dict(), "./save_model/lstm.pkl")
acc = []
count = 10
for step, (b_x, b_y) in enumerate(test_loader):
    if step < count:
        test_output = rnn(b_x)  # (samples, time_step, input_size)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        result = [1 if x == y else 0 for x, y in zip(pred_y, b_y)]
        accuracy = sum(result) / len(result)
        acc.append(accuracy)
    else:
        break
print(np.mean(acc))

# print 10 predictions from test data
# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')
