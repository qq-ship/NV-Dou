import numpy as np
import torch
import os

class miAgent(object):
    def __init__(self, action_num):
        self.action_num = action_num

    def choose_action(self,state,legal_action):

        return int(np.min(legal_action))


class abAgent(object):

    def __init__(self, action_num):


        self.action_num = action_num
        if os.path.exists(r'op_model/ab_rhcp_model/anework.pkl'):
            print("加载抽象模型-----------------------")
            self.anet = torch.load(r'op_model/ab_rhcp_model/anework.pkl')
        else:
            self.anet = miAgent(309)


    def step(self,state,player_id):
        s = np.array(state['obs'])
        action = self.anet.choose_action(np.expand_dims(s,0), state["legal_actions"])
        return int(action)
    def eval_step(self,state,player_id):

        return self.step(state,player_id)
