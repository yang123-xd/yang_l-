import torch
from torch import nn
from d2l import torch as d2l

# 定义一个二则交叉相关
def corr2d(x,k):
    h, w = k.shape
    y = torch.zeros(x.shape[0]-h+1,x.shape[1]- w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i+h, j:j+w] * k).sum()
    return y

x = torch.tensor([[0.0, 1.0, 2.0],[3.0, 4.0, 5.0],[6.0, 7.0, 8.0]])
k = torch.tensor([[0.0, 1.0], [2.0,3.0]])
print(corr2d(x, k))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

x = torch.ones(6,8)
x[:, 2:6] = 0
print(x)

k = torch.tensor([[1.0, -1.0]])
y = corr2d(x, k)
print(y)

print(corr2d(x.t(),k))

# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = x.reshape((1, 1, 6, 8))
Y = y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(20):
    Y_hat = conv2d(X)
    # 均方误差
    l = (Y_hat - Y) ** 2
    # 梯度置零
    conv2d.zero_grad()
    # 反向更新
    l.sum().backward()
    # 迭代卷积核，梯度下降
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    # 每两次输出
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
print(conv2d.weight.data.reshape((1, 2)))


