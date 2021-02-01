import numpy as np
import json
from op_model.rhcp_model.PokerMapping import numpytostr
from op_model.rhcp_model.RHCP import zhudongPut,beidongPut
from op_model.rhcp_model.testCode.dataToType import typeDic
"""
json_card_type = "op_model/rljson/card_type.json"
fp = open(json_card_type)
card_type = json.load(fp)
"""
class RhcpAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): the size of the ouput action space
        '''
        self.action_num = action_num

    @staticmethod
    def step(state,player_id):
        cardstr,one_last,two_last,three_last,legal_card  =  numpytostr(state)
        print("player:", player_id, "手牌 is:", cardstr)
        if 308 not in legal_card:
            putcard = zhudongPut(cardstr)
            #actioncard = typeDic[putcard][0]['ai']
        else:
            upcard = ""
            if one_last=="":
                upcard = two_last
            else:
                upcard = one_last
            putcard = beidongPut(upcard,cardstr)

        putcard = putcard.replace('B', 'R')
        putcard = putcard.replace('L', 'B')

        if putcard is '0':
            putcard = 'pass'

        return putcard
    def eval_step(self,state,player_id):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state,player_id)
