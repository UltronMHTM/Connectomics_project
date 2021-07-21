import torch
import torch.nn as nn
import torch.nn.functional as torchFun
from data_loader.dataLoader import dataLoader
import numpy as np
# from tensorflow.keras.utils import to_categorical

print(torch.__version__ )
print(torch.version.cuda)
print(torch.cuda.is_available())

class test3dCNN(nn.Module):
    def __init__(self):
        super(test3dCNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc = nn.Linear(int(384*384*384/8), 3)
    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x1 = self.layer3(x)
        x1 = self.relu(x1)
        x = torch.add(x, x1)
        x2 = self.layer4(x)
        x2 = self.relu(x2)
        x = torch.add(x, x2)
        x = self.layer5(x)
        x = self.relu(x)
        x = x.view(1,int(384*384*384/8))
        out = self.fc(x)
        return out




def train(trainingData, model, loss_fn, optimizer):
    for batch, (X_raw,X_seg,Y) in enumerate(trainingData):
        X = [[X_seg]]
        X = torch.tensor(X)
        X = X.type(torch.FloatTensor)

        Y = [Y]
        Y = torch.tensor(Y)
        Y = Y.long()
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        print(pred,"--",Y)
        loss = loss_fn(pred,Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        print(f"loss: {loss_val}")

epochNum = 10
for t in range(epochNum):
    print(f"Epoch {t+1}\n----------")
    train(totalData, net, loss_fn, optimizer)
