import numpy as np
import torch
import os
from game_model.PokerMapping import cardcouple
from op_model.rhcp_shang_model.PokerMapping import numpytostr,rltorh,rhtorl
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
from op_model_test.rhcp_shang_model.env import Env as CEnv
from op_model_test.rhcp_shang_model.card import Card
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}
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

        cardstr, one_last, two_last, three_last, legal_card = numpytostr(state)

        curreny_hand = rltorh(cardstr)

        values = CEnv.get_cards_ass(Card.char2color(curreny_hand))

        type,mymax,num = values[0],values[1],values[2]

        actionzong = cardcouple(type,mymax,num)

        for ca in actionzong:
            print("组合牌面：：：：",ACTION_ID_TO_STR[ca])

        hou_legal_action  = []
        if 308 in  state["legal_actions"]:
            hou_legal_action.append(308)
            for a in actionzong:
                if a in state["legal_actions"]:
                    hou_legal_action.append(a)
        else:
            hou_legal_action = actionzong

        hou_legal_action = sorted(hou_legal_action)

        #action = self.anet.choose_action(np.expand_dims(s, 0),hou_legal_action)

        action = hou_legal_action[0]

        print("---玩家:#", player_id, "#手牌是:", cardstr)
        print("---出牌:", ACTION_ID_TO_STR[action], "id:", action)
        return int(action)
    def eval_step(self,state,player_id):

        return self.step(state,player_id)
