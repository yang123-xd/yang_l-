import torch
from torch import nn
from d2l import torch as d2l
import d2l_zh as d2ll

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std= 0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

d2ll.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)