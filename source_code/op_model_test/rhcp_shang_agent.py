from op_model_test.rhcp_shang_model.card import Card
from op_model_test.rhcp_shang_model.utils import to_char, to_value
from op_model_test.rhcp_shang_model.env import Env as CEnv
from op_model_test.rhcp_shang_model.PokerMapping import numpytostr,rltorh,rhtorl

class RhcpShangAgent():
    ''' A random agent. Random agents is for running toy examples on the card games
       '''

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): the size of the ouput action space
        '''
        self.action_num = action_num
        self.num = 0



    def step(self,state, player_id):

        cardstr, one_last, two_last, three_last, legal_card = numpytostr(state)
        #print("state is:", state['obs'],"one_last:",one_last,"two_last is:",two_last,"three_last:",three_last)
        #print("player:", player_id, "手牌 is:", cardstr)
        curreny_hand = rltorh(cardstr)
        one_last_hand = rltorh(one_last)
        two_last_hand = rltorh(two_last)
        three_last = rltorh(three_last)

        upcard = []
        if self.num ==0:
            if len(two_last_hand) == 0:
                upcard = three_last
            else:
                upcard = two_last_hand
        else:
            if len(one_last_hand) != 0:
                upcard = one_last_hand
            elif len(one_last_hand) == 0 and len(two_last_hand) != 0:
                upcard = two_last_hand

        putcard = to_char(CEnv.step_auto_static(Card.char2color(curreny_hand), to_value(upcard)))
        putcard = rhtorl(putcard)
        values = CEnv.get_cards_value(Card.char2color(curreny_hand))
        print("---玩家:#", player_id,"#手牌是:", cardstr,"上一轮出牌：",upcard,"values:")
        print("---出牌:",putcard,"id:",putcard)
        self.num = self.num  + 1

        return putcard

    def eval_step(self, state, player_id):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state, player_id)

    def put_card(self,curreny_hand,last_card):
        intention = to_char(CEnv.step_auto_static(Card.char2color(curreny_hand), to_value(last_card)))
        return intention