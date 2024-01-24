import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from d2l import torch as d2l
import torch
import torch.nn as nn
import csv

# 数据路径
path = "/home/jishengpeng/美赛的模拟练习"

# 加载比特币和黄金价格数据
BCHAIN_MKPRU = pd.read_csv(path + "/BCHAIN-MKPRU.csv", dtype={"Date": np.str, "Value": np.float64})
LBMA_GOLD = pd.read_csv(path + "/LBMA-GOLD.csv", dtype={"Date": np.str, "Value": np.float64})
Data = pd.read_csv(path + "/C题处理后的中间文件2.csv")

# 将日期转换为自然数
def to_timestamp(date):
    return int(time.mktime(time.strptime(date, "%m/%d/%y")))

# 将日期转换为自然数并标准化数据集
start_timestamp = to_timestamp(Data.iloc[0, 0])
for i in range(Data.shape[0]):
    Data.iloc[i, 0] = (to_timestamp(Data.iloc[i, 0]) - start_timestamp) / 86400
print(Data)

# 模型参数
batch_size = 1
start_input = 30
input_size = Data.shape[0]
hidden_size = 20
output_size = 1
layers_size = 3
lr = 10
num_epochs = 1000

# GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_size):
        super().__init__()
        self.GRU_layer = nn.GRU(input_size, hidden_size, layers_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.GRU_layer(x)
        x = self.linear(x)
        return x

# 设备设置为CUDA
device = torch.device("cuda")

# 初始化GRU模型、损失函数和优化器
gru = GRUModel(30, hidden_size, output_size, layers_size).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(gru.parameters(), lr)

# 提取比特币价格数据
ji = np.array(Data.iloc[0:input_size, 3].dropna())
input_size = ji.shape[0] - 2

# 准备训练数据
trainB_x = torch.from_numpy(ji[input_size - 30:input_size].reshape(-1, batch_size, 30)).to(torch.float32).to(device)
trainB_y = torch.from_numpy(ji[input_size].reshape(-1, batch_size, output_size)).to(torch.float32).to(device)

losses = []

# 训练循环
for epoch in range(num_epochs):
    output = gru(trainB_x).to(device)
    loss = criterion(output, trainB_y)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss" + str(epoch) + ":", loss.item())

# 预测，以比特币为例
pred_x_train = torch.from_numpy(ji[input_size - 29:input_size + 1]).reshape(-1, 1, 30).to(torch.float32).to(device)
pred_y_train = gru(pred_x_train).to(device)
print("prediction:", pred_y_train.item())
print("actual:", ji[input_size + 1])

# 可视化训练损失
plt.plot(losses)

# 可视化预测和实际值
plt.plot(pred_y_train.cpu().detach().numpy().flatten(), label='Predicted')
plt.plot(ji[input_size + 1], label='Actual')
plt.legend()

# 输出预测和实际值到CSV文件
f = open(path + '/周期lstm黄金预测1000版本.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["实际价格", "预测价格"])
for i in range(0, input_size - 29):
    tmp = []
    tmp.append(ji[i + 1])  # 实际价格
    tmp.append(round(predictions[i], 2))  # 预测价格
    csv_writer.writerow(tmp)
f.close()
