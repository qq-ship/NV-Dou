#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
from game_model.PokerMapping import numpytostr,rltorh,rhtorl,CardsLeixing,leixingstr
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}
import math

class Net(nn.Module):
    def __init__(self, action_num=2, state_shape=None, mlp_layers=None):
        super(Net, self).__init__()

        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2),
                                   nn.Tanh())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
                                   nn.Tanh())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
                                   nn.Tanh(), nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(11520, self.action_num, bias=True)
        self.fc00 = nn.Linear(11520,15, bias=True)
        self.fc1 = nn.Linear(11520, 1, bias=True)

        self.loss = nn.CrossEntropyLoss()

        self.distribution = torch.distributions.Categorical

    def forward(self, s):
        s = s.view(s.size(0),1,30,15)

        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)

        s = s.view(s.size(0), -1)

        action_logits = F.softmax(self.fc(s))
        class_logits = F.softmax(self.fc00(s))
        state_values = F.tanh(self.fc1(s))

        return action_logits,class_logits,state_values

    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)

    def shaixuan(self,legal_action):
        legal_action_str = [ACTION_ID_TO_STR[ac] for ac in legal_action]
        delid = []
        rs_legal_action = []
        for i in range(len(legal_action_str)):

            for j in range(i+1,len(legal_action_str)):
                if legal_action_str[i] in legal_action_str[j] and '*' not in legal_action_str[i]:
                    if legal_action_str[i] not in range(238,269) and legal_action_str[i] not in range(281,294):
                        delid.append(legal_action[i])
                        break


        for car in legal_action:
            if car not in delid:
                rs_legal_action.append(car)

        return rs_legal_action


    def choose_action(self,state,legal_action):
        self.eval()
        legal_action = sorted(legal_action)
        prob,class_logits,_ = self.forward(self.v_wrap(state))
        prob = prob.data.numpy()[0]
        class_logits = class_logits.data.numpy()[0]

        #legal_action = self.shaixuan(legal_action)

        legal_dis = []
        for i in legal_action:
            legal_dis.append(prob[i])
        legal_dis = np.array(legal_dis)
        kk = np.sum(legal_dis)
        legal_dis = legal_dis/kk
        ran = random.random()
        if ran <= 0.1:
            action = random.choices(legal_action, k=1)
        else:
            #action = random.choices(legal_action, weights=legal_dis)
            index = np.argmax(legal_dis)
            #print("index ------------:",index)
            action = np.array([legal_action[index]])
            #print("action ------------:",action)
        #action = random.choices(legal_action, weights=legal_dis)

        return action[0]

    def p_choose_action(self, state, legal_action):
        self.eval()

        legal_action = sorted(legal_action)
        prob, _, features = self.forward(self.v_wrap(state))
        prob = prob.data.numpy()[0]

        action = np.argmax(prob)

        if action in legal_action:
            return action
        else:
            kk = 400
            tmp =legal_action[0]
            for ac in legal_action:
                abs(ac - action) < kk
                kk = abs(ac - action)
                tmp = ac
            action = tmp
            return action

    def t_choose_action(self, state, legal_action):
        self.eval()

        legal_action = sorted(legal_action)
        prob, _, features = self.forward(self.v_wrap(state))

        return prob

    def loss_func(self, s, a,value_target):

        self.train()
        prob,class_prob,value = self.forward(s)
        value = value.squeeze()
        td_error = value_target - value
        critic_loss = td_error.pow(2)

        m = self.distribution(prob)
        m01 = self.distribution(class_prob)
        log_pob = m.log_prob(a)

        exp_v = log_pob * td_error.detach().squeeze()
        actor_loss = -exp_v
        loss = (critic_loss + actor_loss).mean()
        return loss

    def p_loss_func(self, s, a):
        self.train()
        prob,class_prob,value = self.forward(s)

        eval_action_probs = a

        ce_loss = - (eval_action_probs * prob).sum(dim=-1).mean()

        return ce_loss
