import torch
from d2l import torch as d2l

# Relu函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

# PRelu函数
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))


# sigmoid 函数
z = torch.sigmoid(x)
d2l.plot(x.detach(), z.detach(), 'z', 'sigmoid(x)', figsize=(5, 2.5))


# 清除以前的梯度
x.grad.data.zero_()
# 返回梯度值
z.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))


# Tanh函数
k = torch.tanh(x)
d2l.plot(x.detach(), k.detach(), 'k', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.show()

# 清除以前的梯度
x.grad.data.zero_()
k.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show()