import rlcard
import os
import torch
import torch.multiprocessing as mp
from net_work.a3c_neural_network import Net,SNet
from op_model.random_agent import RandomAgent
from op_model.min_agent import MinAgent
from logger.logger import Logger
from game_model.normal.normal import mean,std
import numpy as np
import math
import sys
import datetime
from op_model.rhcp_shang_agent import RhcpShangAgent
from game_model.PokerMapping import numpytostr,rltorh,rhtorl,CardsLeixing,leixingstr,cardcouple,duotodan
from op_model.ab_rhcp_agent import abAgent
import random
from rlcard.utils.utils import remove_illegal
from op_model.rhcp_model.RHCP import getMaxPoker
from op_model.rhcp_shang_model.card import Card
from op_model.rhcp_shang_model.utils import to_char, to_value
from op_model.rhcp_shang_model.env import Env as CEnv
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}

os.environ["OMP_NUM_THREADS"] = "1"
UPDATE_GLOBAL_ITER = 10

random_agent = RandomAgent(309)
min_agent = MinAgent(309)
rhcp_shang_agent = RhcpShangAgent(309)
rhcp_shang_agent01 = RhcpShangAgent(309)
ab_agent = abAgent(309)

nowTime = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
path = "out_put/test/a3c/"+nowTime
os.makedirs(path)

log_path = path +'/'+'log.txt'
csv_path = path +'/'+'performance.csv'
logger = Logger(xlabel='epsoide', ylabel='landlord',zlabel='farmernext',mlabel='farmerup', legend='a3c on Dou Dizhu', log_path=log_path,csv_path=csv_path)

evaluate_num = 1000

save_plot_every = 1000
eval_every_eposide = 1000
save_mode_every_eposide = 10000
clear_memory_every_eposide = 100000
train_step = 1

class Memory(object):
    def __init__(self,size):
        self.memory = []
        self.size = size
    def addmemory(self,mem):
        if len(self.memory)>self.size:
            self.memory.pop(0)
        self.memory.append(mem)

    def sample(self,num):
        obs = []
        act = []
        if len(self.memory) <= num:
            rs = self.memory
            for ts in rs:
                obs.append(ts['obs'])
                act.append(ts['action'])

            return obs,act
        else:
            rs = random.sample(self.memory, num)
            for ts in rs:
                obs.append(ts['obs'])
                act.append(ts['action'])
            return obs,act

class A3cAgent(object):

    def __init__(self,action_num,lnet,num):
        self.id = num
        self.action_num = action_num
        self.lnet = lnet
        self.num = 0

        self.ran = random.random()
        if os.path.exists(r'game_model/model/w0snet' + str(self.id) + '.pkl'):
            print("加载——————snet 模型---------------w0snet" + str(self.id))
            self.snet = torch.load(r'game_model/model/w0snet' + str(self.id) + '.pkl')

        else:
            print("新建******snet 模型-----------------------")
            self.snet = SNet(309, [6, 5, 15], [512, 1024, 2048, 1024, 512]).cuda()

        self.pop = torch.optim.Adam(self.snet.parameters(), lr=0.0005)
        self.mem = Memory(1e5)
        self.num = 0
        self.action_values = [[50,1] for i in range(309)]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent_probs = []

    def updateMode(self,newlnet):
        self.lnet = newlnet
        self.agent_probs = []
        self.ran = random.random()
        #print("self.ran",self.ran)

    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def step(self,state,player_id):
        s = np.array(state['obs'])
        leagl_actions = state["legal_actions"]
        #cardstr,other_two_action,one_last, two_last, three_last, legal_card = numpytostr(state)

        if self.ran < -0.5:
            #print("zhi xing jiangducelue")
            probs = self.snet.t_choose_action(np.expand_dims(s,0), state["legal_actions"]).detach().cpu().numpy()
            self.agent_probs.append(probs)
            probs = remove_illegal(probs[0], leagl_actions)
            action = np.random.choice(len(probs), p=probs)
            # print("player:", player_id, "手牌 is:", cardstr, "上一局:",0, "出牌:", ACTION_ID_TO_STR[action])
            return int(action)

        else:
            #print("zhi xing zhi qianghua")
            probs = self.lnet.t_choose_action(np.expand_dims(s, 0),state["legal_actions"]).detach().cpu().numpy()
            self.agent_probs.append(probs)

            probs = remove_illegal(probs[0], leagl_actions)
            action = np.random.choice(len(probs), p=probs)

            memdic = {}
            memdic['obs'] = np.expand_dims(s, 0)
            memdic['action'] = action
            self.mem.addmemory(memdic)

            # print("player:", player_id, "手牌 is:", cardstr, "上一局:",0, "出牌:", ACTION_ID_TO_STR[action])
            return int(action)


    def eval_step(self,state,player_id):
        s = np.array(state['obs'])
        leagl_actions = state["legal_actions"]
        #cardstr, other_two_action, one_last, two_last, three_last, legal_card = numpytostr(state)
        #probs = self.lnet.t_choose_action(np.expand_dims(self.normalizer.normalize(s), 0),state["legal_actions"]).detach().cpu().numpy()

        probs = self.lnet.t_choose_action(np.expand_dims(s, 0),state["legal_actions"]).detach().cpu().numpy()
        probs = remove_illegal(probs[0], leagl_actions)
        action = np.random.choice(len(probs), p=probs)
        #print("player:", player_id, "手牌 is:", cardstr, "上一局:",0, "出牌:", ACTION_ID_TO_STR[action])
        return int(action)

    def train_sl(self):
        bs, ba = self.mem.sample(32)

        bs = self.v_wrap(np.array(bs))
        ba = self.v_wrap(np.array(ba))
        loss = self.snet.loss_func(bs,ba)

        self.pop.zero_grad()
        loss.backward()
        self.pop.step()

class TWorker():
    def __init__(self, gnet,opt, global_ep, global_ep_r, res_queue, name,max_episode,update_global_iteration, gamma):
        super(TWorker, self).__init__()
        self.w_id = name
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.env = rlcard.make('doudizhu')
        self.eval_env = rlcard.make('doudizhu')
        self.state_shape = self.env.state_shape
        self.actiom_num = self.env.action_num
        self.max_episode = max_episode

        self.lnet = torch.load(r'game_model/model/w0nework.pkl')
        self.lnet01 = torch.load(r'game_model/model/w0nework01.pkl')
        self.lnet02 = torch.load(r'game_model/model/w0nework02.pkl')


        self.agent = A3cAgent(self.actiom_num,self.lnet,0)
        self.agent01 = A3cAgent(self.actiom_num,self.lnet01,1)
        self.agent02 = A3cAgent(self.actiom_num,self.lnet02,2)

        self.rr = 0
        self.rr01 = 0
        self.rr02 = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def test(self):
        normal_size = 0
        episode = 0
        sum_reward = [0,0,0]
        while episode < self.max_episode:

            #定义程序开始时间
            m_s = datetime.datetime.now()

            self.lnet.remove_noise()
            self.lnet01.remove_noise()
            self.lnet02.remove_noise()

            self.agent.updateMode(self.lnet)
            self.agent01.updateMode(self.lnet01)
            self.agent02.updateMode(self.lnet02)

            self.agent.num = 0
            self.eval_env.set_agents([self.agent,rhcp_shang_agent,rhcp_shang_agent01])

            reward = 0
            reward01 = 0
            reward02 = 0

            for i in range(evaluate_num):
                # self.progress_bar(i+1,500)
                # print("***********************",i,"************************************",reward)
                rhcp_shang_agent.num = 0
                rhcp_shang_agent01.num = 0
                guiji, pay_off = self.eval_env.run(is_training=False, seed=None)
                reward = reward + pay_off[0]

            self.eval_env.set_agents([rhcp_shang_agent, self.agent01, self.agent02])

            for i in range(evaluate_num):
                # self.progress_bar(i+1,500)
                # print("***********************",i,"************************************",reward)
                rhcp_shang_agent.num = 0
                rhcp_shang_agent01.num = 0

                guiji, pay_off = self.eval_env.run(is_training=False, seed=None)

                if len(guiji[1]) == len(guiji[2]):
                    reward02 = reward02 + pay_off[2]
                else:
                    reward01 = reward01 + pay_off[1]

            self.rr = float(reward) / evaluate_num
            self.rr01 = float(reward01) / evaluate_num
            self.rr02 = float(reward02) / evaluate_num

            logger.add_point(x=episode, y=float(reward) / evaluate_num,
                             z=float(reward01) / evaluate_num, m=float(reward02) / evaluate_num)
            self.res_queue.put(self.rr)



            m_e = datetime.datetime.now()
            sec = m_e - m_s
            print("\n", "worker:", self.name, "the test is:", episode,"cost is:", sec.seconds,"reward is:", self.rr, "::", self.rr01, "::",self.rr02)
            episode = episode + 1
