import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from data_loader.dataLoader import *
from tensorboardX import SummaryWriter
import time

batchSize = 2
inputSize = 224
startEpoch = 450
class Dense2Dblock(nn.Module):
    def __init__(self, in_clannelNum):
        super(Dense2Dblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_clannelNum, out_channels=in_clannelNum, kernel_size=3,stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(in_channels=in_clannelNum, out_channels=in_clannelNum, kernel_size=3,stride=1, padding=1).cuda()
        self.conv3 = nn.Conv2d(in_channels=in_clannelNum*2, out_channels=in_clannelNum*2, kernel_size=3,stride=1, padding=1).cuda()
        self.conv4 = nn.Conv2d(in_channels=in_clannelNum*4, out_channels=in_clannelNum, kernel_size=3,stride=1, padding=1).cuda()
        self.relu = nn.ReLU().cuda()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()
    def forward(self, input):
        conv1_out = self.conv1(input)
        conv1_out =self.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)
        conv12_add = torch.cat((conv1_out, conv2_out),dim=1)
        conv3_out = self.conv3(conv12_add)
        conv3_out = self.relu(conv3_out)
        conv123_add = torch.cat((conv12_add, conv3_out), dim=1)
        conv4_out = self.conv4(conv123_add)
        conv4_out = self.relu(conv4_out)
        maxpooling = self.maxpool(conv4_out)
        return maxpooling

class Dense3Dblock(nn.Module):
    def __init__(self, in_clannelNum=1):
        super(Dense3Dblock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_clannelNum, out_channels=in_clannelNum, kernel_size=3,stride=1, padding=1).cuda()
        self.conv2 = nn.Conv3d(in_channels=in_clannelNum, out_channels=in_clannelNum, kernel_size=3,stride=1, padding=1).cuda()
        self.conv3 = nn.Conv3d(in_channels=in_clannelNum*2, out_channels=in_clannelNum*2, kernel_size=3,stride=1, padding=1).cuda()
        self.conv4 = nn.Conv3d(in_channels=in_clannelNum*4, out_channels=in_clannelNum, kernel_size=3,stride=1, padding=1).cuda()
        self.relu = nn.ReLU().cuda()
    def forward(self, input):
        input = self.conv1(input)
        conv1_out = self.relu(input)
        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)
        conv12_add = torch.cat((conv1_out, conv2_out),dim=1)
        conv3_out = self.conv3(conv12_add)
        conv3_out = self.relu(conv3_out)
        conv123_add = torch.cat((conv12_add, conv3_out), dim=1)
        conv4_out = self.conv4(conv123_add)
        conv4_out = self.relu(conv4_out)
        return conv4_out

class FeatureExtractionNet_inputChannel(nn.Module):
    def __init__(self):
        super(FeatureExtractionNet_inputChannel,self).__init__()
        self.model = nn.Sequential().cuda()
        self.model.add_module("conv_First",torch.nn.Conv2d(in_channels=inputSize, out_channels=inputSize, kernel_size=3, stride=1, padding=1))
        self.model.add_module("DenseBlock2D_1",Dense2Dblock(inputSize))
        # self.model.add_module("DenseBlock2D_2", Dense2Dblock(inputSize))
    def forward(self, input):
        features = self.model(input)
        return features

class FeatureExtractionNet(nn.Module):
    def __init__(self):
        super(FeatureExtractionNet, self).__init__()
        self.rawFeatureEx = FeatureExtractionNet_inputChannel()
        self.segFeatureEx = FeatureExtractionNet_inputChannel()
        self.convConcat = nn.Conv2d(in_channels=inputSize*2, out_channels=inputSize, kernel_size=3,stride=1, padding=1).cuda()
        self.DenseBlock2D_3 = Dense2Dblock(inputSize)
        self.DenseBlock3D_1 = Dense3Dblock()
        self.maxpool3D_1 = nn.MaxPool3d(kernel_size=2, stride=2).cuda()
        self.DenseBlock3D_2 = Dense3Dblock()
        self.maxpool3D_2 = nn.MaxPool3d(kernel_size=2, stride=2).cuda()
        self.DenseBlock3D_3 = Dense3Dblock()

    def forward(self, inputRaw, inputSeg):
        outputRaw = self.rawFeatureEx(inputRaw)
        outputSeg = self.segFeatureEx(inputSeg)
        output = torch.cat((outputRaw, outputSeg),dim=1)
        output = self.convConcat(output)
        output = self.DenseBlock2D_3(output)
        output = output.unsqueeze(1).to(device)
        output = self.DenseBlock3D_1(output)
        output = self.maxpool3D_1(output)
        output = self.DenseBlock3D_2(output)
        output = self.maxpool3D_2(output)
        output = self.DenseBlock3D_3(output)
        return output

class FeatureExtractionNet_train(nn.Module):
    def __init__(self):
        super(FeatureExtractionNet_train, self).__init__()
        self.FeatureEx = FeatureExtractionNet()
        self.fc1 = nn.Linear(14*14*56,4096).cuda()
        self.fc2 = nn.Linear(4096, 4096).cuda()
        self.fc3 = nn.Linear(4096, 3).cuda()

    def forward(self, inputRaw, inputSeg):
        output = self.FeatureEx(inputRaw, inputSeg)
        output = output.view(-1, 14*14*56)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def train(trainingData, model, loss_fn, optimizer):
    # sumLoss = torch.zeros(1).squeeze().to(device)
    for batch, data in enumerate(trainingData):
        X_seg = data[0]
        X_raw = data[1]
        Y = data[2]
        X_seg = torch.tensor(X_seg).cuda()
        X_seg = X_seg.type(torch.cuda.FloatTensor).to(device)
        X_raw = torch.tensor(X_raw).cuda()
        X_raw = X_raw.type(torch.cuda.FloatTensor).to(device)
        Y = torch.tensor(Y).cuda()
        Y = Y.type(torch.cuda.LongTensor).to(device)
        X_seg, X_raw, Y = X_seg.to(device), X_raw.to(device), Y.to(device)
        pred = model(X_raw, X_seg)
        # print(pred,"--",Y)
        # loss = loss_fn(pred,Y).cuda()
        loss = loss_fn(pred, Y).cuda()
        # sumLoss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_val = loss.item()
        # print(f"loss: {loss_val}")

    # print("training loss = {}".format((sumLoss/len(trainingData)).cpu().detach().data.numpy()))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
net = FeatureExtractionNet_train().cuda()
net = net.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1) #1e-3
scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.95)
print(net)


# merge: 2, split: 1, correct: 0
tran = transforms.ToTensor()
dataloader = dataLoader("../DataGeneration/generatedDataset224_300.h5")
positive = list(zip(list(tran(sample).detach().numpy() for sample in np.array(dataloader.pos_sample_raw)),
                    dataloader.pos_sample_seg_b,
                    np.zeros(len(dataloader.pos_sample_seg_b))))
mergeError = list(zip(list(tran(sample).detach().numpy() for sample in np.array(dataloader.neg_m_sample_raw)),
                      dataloader.neg_m_sample_seg_b,
                      2*np.ones(len(dataloader.neg_m_sample_seg_b))))
splitError = list(zip(list(tran(sample).detach().numpy() for sample in np.array(dataloader.neg_s_sample_raw)),
                      dataloader.neg_s_sample_seg_b,
                      np.ones(len(dataloader.neg_s_sample_seg_b))))

totalData = np.vstack((positive,mergeError))
totalData = np.vstack((totalData,splitError))
print(totalData.shape)
np.random.seed(1)
np.random.shuffle(totalData)

trainSet = totalData[0:int(len(totalData)*0.7)]
testSet = totalData[int(len(totalData)*0.7):-1]
trainSet = dataSet(trainSet[:,0],trainSet[:,1],trainSet[:,2])
testSet = dataSet(testSet[:,0],testSet[:,1],testSet[:,2])

trainLoader = DataLoader(trainSet, batch_size=batchSize,shuffle=True)
testLoader = DataLoader(testSet, batch_size=batchSize,shuffle=True)

net.load_state_dict(torch.load("models_test/FeatureExtractionNet_train{}.pth".format(startEpoch)))
writer = SummaryWriter("../Results/r1_test")
time_start = time.time()
epochNum = 1000
previousAcc = 0
for t in range(startEpoch, startEpoch+epochNum):
    net = net.train()
    print(f"Epoch {t + 1}\n----------")
    train(trainLoader, net, loss_fn, optimizer)
    if t % 10 == 0 and t != 0:
        torch.save(obj=net.state_dict(), f="models_test/FeatureExtractionNet_train{}.pth".format(t))
    net = net.eval()
    if t % 5 == 0:
        correct = torch.zeros(1).squeeze().to(device)
        total = torch.zeros(1).squeeze().to(device)
        # sumLoss_t = torch.zeros(1).squeeze().to(device)
        for i, data in enumerate(testLoader):
            X_seg = data[0]
            X_raw = data[1]
            Y = data[2]
            X_seg = torch.tensor(X_seg).cuda()
            X_seg = X_seg.type(torch.cuda.FloatTensor).to(device)
            X_raw = torch.tensor(X_raw).cuda()
            X_raw = X_raw.type(torch.cuda.FloatTensor).to(device)
            Y = torch.tensor(Y).cuda()
            Y = Y.type(torch.cuda.LongTensor).to(device)
            output = net(X_raw, X_seg)
            loss_t = loss_fn(output, Y)
            loss_t = loss_t.to(device)
            # sumLoss_t += loss_t
            prediction = torch.argmax(output, 1)
            correct += (prediction == Y).sum().float()
            total += len(Y)
        accuracy = (correct / total).cpu().detach().data.numpy()
        acc_str = 'Accuracy: %f' % (accuracy)
        if accuracy > previousAcc and t != startEpoch:
            previousAcc = accuracy
            writer.add_scalar("Accuracy increasing", accuracy, t)
        # print("test loss = {}".format((sumLoss_t/len(testSet)).cpu().detach().data.numpy()))

        print(acc_str)
        # writer.add_scalar("test loss", sumLoss_t / len(testSet), t)
        writer.add_scalar("Accuracy total", accuracy, t)
    time_end = time.time()
    scheduler.step()
print('Total time cost: {}s'.format(time_end - time_start))
writer.close()


