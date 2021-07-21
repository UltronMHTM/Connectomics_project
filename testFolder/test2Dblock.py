import torch
import torch.nn as nn
import torch.nn.functional as torchFun
from data_loader.dataLoader import dataLoader
import numpy as np
from tensorboardX import SummaryWriter
# from tensorflow.keras.utils import to_categorical

print(torch.__version__ )
print(torch.version.cuda)
print(torch.cuda.is_available())

class test2dCNN(nn.Module):
    def __init__(self):
        super(test2dCNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,stride=1, padding=1)
        self.layer2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,stride=1, padding=1)
        self.layer3 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,stride=1, padding=1)
        self.layer4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(int(384*384*384/4), 3)
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
        x = x.view(1,int(384*384*384/4))
        out = self.fc(x)
        return out


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
net = test2dCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
print(net)


# merge: 2, split: 1, correct: 0
dataloader = dataLoader("../DataGeneration/generatedDataset.h5")
positive = list(zip(dataloader.pos_sample_raw, dataloader.pos_sample_seg, np.zeros(len(dataloader.pos_sample_seg))))
mergeError = list(zip(dataloader.neg_m_sample_raw, dataloader.neg_m_sample_seg, 2*np.ones(len(dataloader.neg_m_sample_seg))))
splitError = list(zip(dataloader.neg_s_sample_raw, dataloader.neg_s_sample_seg, np.ones(len(dataloader.neg_s_sample_seg))))

totalData = np.vstack((positive,mergeError))
totalData = np.vstack((totalData,splitError))
print(totalData.shape)
np.random.shuffle(list(totalData))

def train(trainingData, model, loss_fn, optimizer):
    for batch, (X_raw,X_seg,Y) in enumerate(trainingData):
        X = [X_seg]
        X = torch.tensor(X)
        X = X.type(torch.FloatTensor)

        Y = [Y]
        Y = torch.tensor(Y)
        Y = Y.cuda.long()
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
writer = SummaryWriter('./Results')

for t in range(epochNum):
    net = net.train()
    print(f"Epoch {t+1}\n----------")
    train(totalData, net, loss_fn, optimizer)
    if t%5==0:
        torch.save(obj=net.state_dict(), f="models/FeatureExtractionNet_train{}.pth".format(t))
    net = net.eval()
    if t%3==0:
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        for i, (X_raw,X_seg,Y) in enumerate(totalData):
            X_raw = torch.tensor([X_raw]).cuda()
            X_seg = torch.tensor([X_seg]).cuda()
            Y = torch.tensor([Y]).cuda()
            output = net(X_raw,X_seg)
            prediction = torch.argmax(output, 1)
            correct += (prediction == Y).sum().float()
            total += len(Y)
        acc_str = 'Accuracy: %f' % ((correct / total).cpu().detach().data.numpy())

    # class Bottleneck(nn.Module):
    #     def __init__(self, channels_in, growth_rate):
    #         super().__init__()
    #         self.growth_rate = growth_rate
    #         self.channels_in = channels_in
    #         self.out_channels_1x1 = 4*self.growth_rate
    #         self.layers = nn.Sequential(nn.BatchNorm2d(num_features=self.channels_in),
    #                                     nn.ReLU(),
    #                                     nn.Conv2d(in_channels=self.channels_in, out_channels=self.out_channels_1x1, kernel_size=1, padding=0,bias=False),
    #                                     nn.BatchNorm2d(num_features=self.out_channels_1x1),
    #                                     nn.ReLU(),
    #                                     nn.Conv2d(in_channels=self.out_channels_1x1, out_channels=self.growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
    #
    #     def forward(self, x):
    #         out = self.layers(x)
    #         out = torch.cat((x, out), dim=1)
    #         return out

    # def Dense3Dblock(in_clannelNum=1, out_channelNum=1, layerNum=4):
    #     block = []
    #     for index in range(layerNum):
    #         block.append(
    #             nn.Conv3d(in_channels=in_clannelNum, out_channels=out_channelNum, kernel_size=3, stride=1,
    #                       padding=1))
    #         block.append(nn.ReLU())
    #     block.append(nn.MaxPool3d(kernel_size=2, stride=2))
    #     return nn.Sequential(*block)

    # block = []
    # for index in range(layerNum):
    #     block.append(
    #         nn.Conv2d(in_channels=in_clannelNum, out_channels=out_channelNum, kernel_size=3, stride=1, padding=1))
    #     block.append(nn.ReLU())
    # block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # return nn.Sequential(*block)
