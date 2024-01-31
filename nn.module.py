import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

x = torch.rand(2, 20)
print(net(x))
# ---------------------------------------------------------------------------------------
class MLP(nn.Module):
    # 用模型参数声明层,这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
net = MLP()
print(net(x))

# -----------------------------------------------------------------------------------------
class Mysequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            self._modules[str(idx)] = module
    def forward(self,x):
        for block in self._modules.values():
            x = block(x)
        return x

net = Mysequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(net(x))

# --------------------------------------------------------------------------------------
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=True)
        self.linear = nn.Linear(20,20)
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(torch.mm(x, self.rand_weight)+1)
        x = self.linear(x)

        while x.abs().sum() > 1:
            x /= 2
        return x.sum()
net = FixedHiddenMLP()
print(net(x))





