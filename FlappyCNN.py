import torch.nn as nn


class FlappyCNN(nn.Module):
    def __init__(self):
        super(FlappyCNN, self).__init__()
        # 开始一层层建立卷积层
        self.conv_1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        # 开始建立全连接层
        self.linear_1 = nn.Sequential(nn.Linear(7*7*64,512), nn.ReLU(inplace=True))
        self.linear_2 = nn.Linear(512, 2)  # 最后输出两个动作的Q值
        self.create_weight()

    def create_weight(self):
        # 开始设置每一层的权重，来避免梯度异常
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):    # 设置卷积层顺序
        output = self.conv_1(data)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = output.view(output.size(0), -1)
        output = self.linear_1(output)
        output = self.linear_2(output)
        return output
