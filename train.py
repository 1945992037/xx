import torch
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torch.nn import init
import torch.optim as optim
from collections import OrderedDict


train_csv=pd.read_csv(r'./fashion-mnist_train.csv')
x=np.array(train_csv.iloc[:,1:])
y=np.array(train_csv.iloc[:,0])
x=torch.FloatTensor(x)
y=torch.FloatTensor(y)


model = nn.Sequential(OrderedDict([
                                   ('fc1',nn.Linear(784,256)),
                                   ('relu1',nn.ReLU()),
                                   ('fc2',nn.Linear(256,64)),
                                   ('relu2',nn.ReLU()),
                                   ('fc3',nn.Linear(64,32)),
                                   ('relu3',nn.ReLU()),
                                   ('fc4',nn.Linear(32,10)),
                                   ('logmax',nn.LogSoftmax(dim=1))]))
#神经网络中所有参数的初始化
for param in model.parameters():
   init.normal_(param,mean=0,std=0.01)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)
torch_dataset = Data.TensorDataset(x,y)

loader = torch.utils.data.DataLoader(
    dataset=torch_dataset,
    batch_size=32,
    shuffle=True,
)

for t in range(200):
    loss_total = 0
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学
        prediction = model(batch_x)
        loss = criterion(prediction,batch_y.long())     # 计算两者的误差
        loss.backward()  # 反向传播 得到梯度
        optimizer.step()  # 更新权重
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss_total+=loss.item()
    print(loss_total/60000)

torch.save(model,'./model')