import torch
from d2l import torch as d2l
import d2l_zh as d2ll
from torch import nn

bath_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(bath_size)
#
num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# 为什莫要乘0.1，其原因是为了使其收敛速度变快，防止梯度爆炸
# w1 = torch.randn(num_inputs, num_hiddens, requires_grad=True)
# b1 = torch.zeros(num_hiddens,requires_grad=True)
# w2 = torch.randn(num_hiddens, num_outputs, requires_grad=True)
# b2 = torch.zeros(num_outputs, requires_grad=True)

params = [w1, b1, w2, b2]

def net(X):
    X = X.reshape((-1, num_inputs))
    H = torch.relu(X@w1 + b1)  # 这里“@”代表矩阵乘法
    return (H@w2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1        
updater = torch.optim.SGD(params, lr=lr)
d2ll.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2ll.plt.show()

