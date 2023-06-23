import torch
from tensorboardX import SummaryWriter


# 设计一个小网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = torch.nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.dense(x)


# 根据小网络实例化一个模型 net
net = Net()
# 创建文件写控制器，将之后的数值以protocol buffer格式写入到logs文件夹中，空的logs文件夹将被自动创建。
writer = SummaryWriter(log_dir='runs')
# 将网络net的结构写到logs里：
data = torch.rand(2, 10)
writer.add_graph(net, input_to_model=(data,))
# 注意：pytorch模型不会记录其输入输出的大小，更不会记录每层输出的尺寸。
#      所以，tensorbaord需要一个假的数据 `data` 来探测网络各层输出大小，并指示输入尺寸。

# 写一个新的数值序列到logs内的文件里，比如sin正弦波。
for i in range(100):
    x = torch.tensor(i / 10, dtype=torch.float)
    y = torch.sin(x)
    # 写入数据的标注指定为 data/sin, 写入数据是y, 当前已迭代的步数是i。
    writer.add_scalar('data/sin', y, i)

writer.close()
