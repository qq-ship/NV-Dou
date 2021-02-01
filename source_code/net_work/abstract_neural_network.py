#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ANet(nn.Module):
    def __init__(self, action_num=2, state_shape=None, mlp_layers=None):
        super(ANet, self).__init__()

        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=2),nn.Tanh())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),nn.Tanh())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),nn.Tanh(),nn.MaxPool2d(kernel_size=2))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),nn.Tanh())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),nn.Tanh(), nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(4928, self.action_num, bias=True)
        self.fc1 = nn.Linear(4928, 1, bias=True)
        self.loss = nn.CrossEntropyLoss()
        self.distribution = torch.distributions.Categorical

    def forward(self, s):
        s = s.view(s.size(0),1,30,15)
        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)
        s = self.conv4(s)
        s = self.conv5(s)

        s = s.view(s.size(0),-1)

        action_logits = F.log_softmax(self.fc(s))
        state_values = F.tanh(self.fc1(s))

        return action_logits,state_values

    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)

    def choose_action(self,state,legal_action):
        self.eval()

        legal_action = sorted(legal_action)
        prob,_ = self.forward(self.v_wrap(state))
        prob = prob.data.numpy()[0]

        taction = np.argmax(prob)


        if taction in legal_action:
            return taction
        else:
            kk = 400
            tmp = legal_action[0]
            for ac in legal_action:
                abs(ac - taction) < kk
                kk = abs(ac - taction)
                tmp = ac
            action = tmp
            return action


    def loss_func(self,s,a):
        self.train()
        prob,_ = self.forward(s)

        loss = self.loss(prob, a)

        yuse = prob.argmax(axis=1)
        muse = a.detach().numpy()
        tong = 0
        for i in range(len(muse)):
            if muse[i] == yuse[i]:
                tong = tong + 1
        return loss,tong