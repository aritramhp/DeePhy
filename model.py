from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Triplet(nn.Module):
    def __init__(self, drop_prob=0.):
        super(Triplet, self).__init__()
        self.drop_prob = drop_prob
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(2,3), stride=(1,1), bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # self.relu1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout2d(.1)
        self.pool1 = nn.AvgPool2d((1,3))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,3), stride=(1,1), bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        # self.relu2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout2d(.1)
        self.pool2 = nn.AvgPool2d((1,3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,3), stride=(1,1), bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        # self.relu3 = nn.Sigmoid()
        self.dropout3 = nn.Dropout2d(.1)
        self.pool3 = nn.AvgPool2d((1,3))

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,3), stride=(1,1), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout2d(.1)
        self.pool4 = nn.AvgPool2d((1,3))

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,3), stride=(1,1), bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout2d(.1)
        self.pool5 = nn.AvgPool2d((1,3))

        # self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,3), stride=(1,1), bias=False)
        # self.bn6 = nn.BatchNorm2d(128)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.dropout6 = nn.Dropout2d(drop_prob)
        # self.maxpool6 = nn.MaxPool2d((1,3))

        # self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,3), stride=(1,1), bias=False)
        # self.bn7 = nn.BatchNorm2d(64)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.dropout7 = nn.Dropout2d(drop_prob)
        # self.maxpool7 = nn.MaxPool2d((1,3))

        self.fc1 = nn.Linear(512, 64)
        # self.fc1 = nn.Linear(58368, 512)
        self.bnf1 = nn.BatchNorm1d(64)
        self.reluf1 = nn.ReLU(inplace=True)
        # self.reluf1 = nn.Sigmoid()
        # self.dropoutf1 = nn.Dropout(drop_prob)        

        self.fc2 = nn.Linear(64, 3)
        # self.bnf2 = nn.BatchNorm1d(3)
        # self.reluf2 = nn.ReLU(inplace=True)
        # self.dropoutf2 = nn.Dropout(.25)
        
        # self.fc2 = nn.Linear(32, num_classes)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, A, B, C):
        # print(A.shape)
        out = torch.cat([A,B,C], dim=2)
        # print(out.shape)
        # exit()
        out = self.pool1(self.dropout1(self.relu1(self.bn1(self.conv1(out)))))
        out = self.pool2(self.dropout2(self.relu2(self.bn2(self.conv2(out)))))
        out = self.pool3(self.dropout3(self.relu3(self.bn3(self.conv3(out)))))
        out = self.pool4(self.dropout4(self.relu4(self.bn4(self.conv4(out)))))
        out = self.pool5(self.dropout5(self.relu5(self.bn5(self.conv5(out)))))
        # out = self.maxpool6(self.dropout6(self.relu6(self.bn6(self.conv6(out)))))
        # out = self.maxpool7(self.dropout7(self.relu7(self.bn7(self.conv7(out)))))
        # out = self.dropout1(self.relu1(self.bn1(self.conv1(out))))
        # out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        # out = self.dropout3(self.relu3(self.bn3(self.conv3(out))))
        # out = self.dropout4(self.relu4(self.bn4(self.conv4(out))))

        # out = self.avgpool1(self.relu1(self.bn1(self.conv1(out))))
        # out = self.avgpool2(self.relu2(self.bn2(self.conv2(out))))
        # out = self.avgpool3(self.relu3(self.bn3(self.conv3(out))))
        # out = self.avgpool4(self.relu4(self.bn4(self.conv4(out))))
        # out = self.avgpool5(self.relu5(self.bn5(self.conv5(out))))
        # out = self.avgpool6(self.relu6(self.bn6(self.conv6(out))))
        # out = self.avgpool7(self.relu7(self.bn7(self.conv7(out))))

        # print(out.shape)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        # exit()
        out = self.reluf1(self.bnf1(self.fc1(out)))
        # out = self.dropoutf2(self.reluf2(self.bnf2(self.fc2(out))))
        out = self.fc2(out)

        # out = self.softmax(out)
        return out
    

