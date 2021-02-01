import numpy as np
import torch
from net_work.a3c_neural_network备份01 import Net

class A3cFarmersAgent(object):
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
        self.lnet = torch.load('op_model/a3c- farmers-model/nework.pkl')


    def step(self,state,player_id):

        s = np.array(state['obs'])
        action = self.lnet.choose_action(np.expand_dims(s, 0),state["legal_actions"])
        return action
    def eval_step(self,state,player_id):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state,player_id)
