#coding=utf-8
import rlcard
from op_model_test.random_agent import RandomAgent
from op_model_test.min_agent  import MinAgent
from op_model_test.a3c_landlord_agent import A3cLandlordAgent
from op_model_test.rhcp_agent import RhcpAgent
from op_model_test.rhcp_shang_agent import RhcpShangAgent
from op_model_test.ab_rhcp_agent import abAgent
from game_model.PokerMapping import numpytostr,rltorh,rhtorl,CardsLeixing,leixingstr
#from op_model_test.rhcp_shang_model.card import Card
#from op_model_test.rhcp_shang_model.utils import to_char, to_value
#from op_model_test.rhcp_shang_model.env import Env as CEnv

#from op_model.a3c_farmers_agent import A3cFarmersAgent
import sys
import math
import datetime

evaluate_num = 100

if __name__ == "__main__":

    m_s = datetime.datetime.now()
    random_agent = RandomAgent(309)
    min_agent = MinAgent(309)
    a3c_andlord_agent = A3cLandlordAgent(309)
    rhcp_agent = RhcpAgent(309)
    rhcp_shang_agent = RhcpShangAgent(309)
    ab_rhcp_agent = abAgent(309)
    #print(rhcp_shang_agent.put_card(['8','8','8','10'],[]))

    eval_env = rlcard.make('doudizhu')
    rhcp_shang_agent.num = 0
    eval_env.set_agents([ab_rhcp_agent,rhcp_shang_agent,rhcp_shang_agent])
    reward = 0

    for i in range(evaluate_num):
        #progress_bar(i,evaluate_num)
        rhcp_shang_agent.num = 0
        guiji, pay_off =  eval_env.run(is_training=False)
        reward = reward + pay_off[0]
        v = 0
        for ts in guiji[0]:
            cardstr, other_card, one_last, two_last, three_last, layed_card = numpytostr(ts[0])
            current_hand = rltorh(cardstr)
            #values = CEnv.get_cards_value(Card.char2color(current_hand))
            #v = v + values[0]
            
        print("***********************************eposid is:",i,"reward is:",reward,"rslist",pay_off[0],"******************************************")
    m_e = datetime.datetime.now()
    sec = m_e - m_s
    print("执行完成,耗时",sec.seconds)
