import numpy as np
from op_model.rhcp_shang_model.PokerMapping import numpytostr,rltorh,rhtorl
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}

class MinAgent(object):
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
        cardstr, one_last, two_last, three_last, legal_card = numpytostr(state)
        #print("player:", player_id, "手牌 is:", cardstr)

        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''

        action = np.min(state['legal_actions'])

        print("---玩家:#", player_id, "#手牌是:", cardstr)
        print("---出牌:", ACTION_ID_TO_STR[action], "id:", action)
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
