import numpy as np
import torch
from op_model.rhcp_shang_model.PokerMapping import numpytostr,rltorh,rhtorl
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}
class A3cLandlordAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): the size of the ouput action space
        '''
        self.action_num = action_num
        """
        self.lnet = Net(309,[6,5,15],[512,1024,2048,1024,512])
        self.lnet.load_state_dict(torch.load('op_model/a3cmodel/nework.pkl'))
        """
        self.lnet = torch.load('op_model/a3c_landlord_model/w0nework.pkl')


    def step(self,state,player_id):

        s = np.array(state['obs'])
        action = self.lnet.choose_action(np.expand_dims(s, 0),state["legal_actions"],0)[0]
        cardstr, one_last, two_last, three_last, legal_card = numpytostr(state)
        print("player:", player_id, "手牌 is:", cardstr)
        #print("出牌:",ACTION_ID_TO_STR[action],"id":action)
        return int(action)
    def eval_step(self,state,player_id):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state,player_id)
