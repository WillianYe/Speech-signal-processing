import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    def __init__(self, class_number):
        super(Net, self).__init__()  # 3*37*13
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(8, 4))  # 6*30*10
        self.batchnormal1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 6*15*5
        self.conv2 = nn.Conv2d(6, 36, kernel_size=(8, 4))  # 36*8*2
        self.batchnormal2 = nn.BatchNorm2d(36)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 36*4*1
        self.flatten = nn.Flatten()  # 144
        self.fc = nn.Linear(144, class_number)  # 10
        self.dropout = nn.Dropout2d(0.01)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.batchnormal1(output)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.batchnormal2(output)
        output = self.relu(output)
        output = self.maxpool2(output)
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.logsoftmax(output)

        return output


# 处理输入数据
filename = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
datas = []
labels = []
for i in range(len(filename)):
    coeffs = sio.loadmat('../DSP_big_project/coeffs' + str(filename[i]) + '.mat')
    delta = sio.loadmat('../DSP_big_project/delta' + str(filename[i]) + '.mat')
    deltaDelta = sio.loadmat('../DSP_big_project/deltaDelta' + str(filename[i]) + '.mat')
    for j in range(len(coeffs) - 3):
        datas.append(np.stack(
            (coeffs['coeffs' + str(j + 1)], delta['delta' + str(j + 1)], deltaDelta['deltaDelta' + str(j + 1)])))
        labels.append(i)
# 设置随机数种子
np.random.seed(0)
torch.manual_seed(0)
# 随机打乱数据
entries = list(zip(datas, labels))
np.random.shuffle(entries)
datas[:], labels[:] = zip(*entries)


class MyDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.datas)


# 区分训练集和测试集
test_datas = torch.Tensor(datas[:int(0.15 * len(datas))])  # 80%训练、20%测试
train_datas = torch.Tensor(datas[int(0.15 * len(datas)):])
test_labels = torch.Tensor(labels[:int(0.15 * len(labels))]).long()  # 80%训练、20%测试
train_labels = torch.Tensor(labels[int(0.15 * len(labels)):]).long()
train_dataset = MyDataset(train_datas, train_labels)
test_dataset = MyDataset(test_datas, test_labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
# 构建模型
model = Net(10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# 开始训练
train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
for epoch in range(30):
    train_correct, train_total, test_correct, test_total = 0, 0, 0, 0
    loss, accuracy = 0, 0
    for datas, labels in train_loader:  # 输入训练集数据
        optimizer.zero_grad()
        outputs = model(datas)  # 输入数据特征参数
        _, predicted = torch.max(outputs.data, 1)  # 计算预测标签
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()  # 预测对了加一
        loss = criterion(outputs, labels)  # 计算训练集损失函数
        loss.backward()  # 梯度下降法调正参数
        optimizer.step()
        accuracy = 100 * torch.true_divide(train_correct, train_total)  # 计算训练集分类精度
    print("epoch:", epoch, " train loss:", loss, " train accuracy:", accuracy)
    train_loss.append(loss)
    train_accuracy.append(accuracy)
    loss, accuracy = 0, 0
    with torch.no_grad():  # 设置为不调整参数
        for datas, labels in test_loader:  # 输入测试集数据
            outputs = model(datas)  # 输入数据特征参数
            _, predicted = torch.max(outputs.data, 1)  # 计算预测标签
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)  # 计算测试集损失函数
            accuracy = 100 * torch.true_divide(test_correct, test_total)  # 计算测试集分类精度
        print("test loss:", loss, " test accuracy:", accuracy)
    test_loss.append(loss)
    test_accuracy.append(accuracy)
# 画图显示结果
plt.figure()
p1, = plt.plot(train_loss, 'r')
p2, = plt.plot(train_accuracy, 'b')
p3, = plt.plot(test_loss, 'g')
p4, = plt.plot(test_accuracy, 'y')
plt.legend([p1, p2, p3, p4], ["train loss", "train accuracy", "test loss", "test accuracy"], loc='center right')
plt.show()
