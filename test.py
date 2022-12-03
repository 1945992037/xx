import torch
import pandas as pd
import torch.utils.data as Data
import numpy as np
import csv
test_csv=pd.read_csv(r'./fashion-mnist_test_data.csv')
id=np.array(test_csv.iloc[:,0])
id=torch.FloatTensor(id)
x=np.array(test_csv.iloc[:,1:])
x=torch.FloatTensor(x)
test_dataset = Data.TensorDataset(x,id)

loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
)
y_hat=[]
model = torch.load('./model')
for batch_x,batch_id in loader:
    prediction = model(batch_x)
    y_hat.append(prediction.argmax(1))

result=[]
for t in y_hat:
    for i in t:
        result.append(i.item())

with open("./result.csv",'w',newline='') as f:
    for i,hat in enumerate(result):
        row = [r'{}.jpg'.format(i),hat]
        write = csv.writer(f)
        write.writerow(row)
