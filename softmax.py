import torch
from torch import nn
from d2l import torch as d2l
import d2l_zh as d2ll

# 初始化参数
batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 定义权重参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std= 0.001)

net.apply(init_weights)
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 小批量优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 10
# 调用模型进行计算
d2ll.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
d2l.plt.show()

