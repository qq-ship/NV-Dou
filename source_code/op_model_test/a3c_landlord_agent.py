import numpy as np
import torch
from op_model.rhcp_shang_model.PokerMapping import numpytostr,rltorh,rhtorl
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}
from game_model.normal.normal import mean,std

class Normalizer(object):
    def __init__(self):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.state_memory = []
        self.max_size = 1000
        self.length = 1000
    def normalize(self, s):
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

class A3cLandlordAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):

        self.action_num = action_num

        self.lnet = torch.load('op_model_test/a3c_landlord_model/w0nework.pkl')
        self.normalizer = Normalizer()


    def step(self,state,player_id):

        s = np.array(state['obs'])

        zhuaction = self.lnet.choose_action(np.expand_dims(self.normalizer.normalize(s), 0),state["legal_actions"])

        cardstr, one_last, two_last, three_last, legal_card = numpytostr(state)
        print("player:", player_id, "手牌 is:", cardstr)
        print("出牌:",ACTION_ID_TO_STR[zhuaction],"id:",zhuaction)

        return int(zhuaction)


    def eval_step(self,state,player_id):

        return self.step(state,player_id)
