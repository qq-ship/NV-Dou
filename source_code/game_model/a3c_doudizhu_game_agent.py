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
path = "out_put/logs/a3c/"+nowTime
os.makedirs(path)

path01 = "out_put/nets/a3c/"+nowTime
os.makedirs(path01)

log_path = path +'/'+'log.txt'
csv_path = path +'/'+'performance.csv'
logger = Logger(xlabel='epsoide', ylabel='landlord',zlabel='farmernext',mlabel='farmerup', legend='a3c on Dou Dizhu', log_path=log_path,csv_path=csv_path)

evaluate_num = 100

save_plot_every = 1000
eval_every_eposide = 2000
save_mode_every_eposide = 10000
clear_memory_every_eposide = 250000
train_step = 1

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

class RMemory(object):
    def __init__(self, size):
        self.memory = []
        self.size = size

    def addmemory(self, mem):
        if len(self.memory) > self.size:
            self.memory.pop(0)
        self.memory.append(mem)

    def sample(self, num):

        obs = []
        act = []
        val = []
        prob = []

        if len(self.memory) <= num:
            rs = self.memory
            for ts in rs:
                obs.append(ts['obs'])
                act.append(ts['action'])
                val.append(ts['val'])
                prob.append(ts['prob'])

            return obs, act, val, prob
        else:

            rs = random.sample(self.memory, num)
            for ts in rs:

                obs.append(ts['obs'])
                act.append(ts['action'])
                val.append(ts['val'])
                prob.append(ts['prob'])

            return obs, act, val, prob
    def clear(self):
        self.memory = []

class A3cAgent(object):

    def __init__(self,action_num,lnet,num):
        self.id = num
        self.action_num = action_num
        self.lnet = lnet
        self.normalizer = Normalizer()
        self.num = 0

        self.ran = random.random()
        if os.path.exists(r'game_model/model/w0snet' + str(self.id) + '.pkl'):
            print("加载——————snet 模型---------------w0snet" + str(self.id))
            self.snet = torch.load(r'game_model/model/w0snet' + str(self.id) + '.pkl')

        else:
            print("新建******snet 模型-----------------------")
            self.snet = SNet(309, [6, 5, 15], [512, 1024, 2048, 1024, 512]).cuda()

        self.pop = torch.optim.Adam(self.snet.parameters(), lr=0.0005)
        self.mem = Memory(1e4)
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
            #The old average strategy
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
            #self.mem.addmemory(memdic)

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
        bs, ba = self.mem.sample(64)

        bs = self.v_wrap(np.array(bs))
        ba = self.v_wrap(np.array(ba))
        loss = self.snet.loss_func(bs,ba)

        self.pop.zero_grad()
        loss.backward()
        self.pop.step()

class Worker():
    def __init__(self, gnet,opt, global_ep, global_ep_r, res_queue, name,max_episode,update_global_iteration, gamma):
        super(Worker, self).__init__()
        self.w_id = name
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.env = rlcard.make('doudizhu')
        self.eval_env = rlcard.make('doudizhu')
        self.state_shape = self.env.state_shape
        self.actiom_num = self.env.action_num

        self.rmem = RMemory(5e5)
        self.rmem01 = RMemory(5e5)
        self.rmem02 = RMemory(5e5)

        if os.path.exists(r'game_model/model/w0nework.pkl'):
            print("加载——————lnet 模型-----------------------")
            self.lnet = torch.load(r'game_model/model/w0nework.pkl')
        else:
            print("新建******lnet 模型-----------------------")
            self.lnet = Net(self.actiom_num, self.state_shape, [512, 1024, 2048, 1024, 512]).cuda()

        if os.path.exists(r'game_model/model/w0nework01.pkl'):
            print("加载——————lnet01 模型-----------------------")
            self.lnet01 = torch.load(r'game_model/model/w0nework01.pkl')
        else:
            print("新建******lnet01模型-----------------------")
            self.lnet01 = Net(self.actiom_num, self.state_shape, [512, 1024, 2048, 1024, 512]).cuda()

        if os.path.exists(r'game_model/model/w0nework02.pkl'):
            print("加载——————lnet02模型-----------------------")
            self.lnet02 = torch.load(r'game_model/model/w0nework02.pkl')
        else:
            print("新建******lnet02模型-----------------------")
            self.lnet02 = Net(self.actiom_num, self.state_shape, [512, 1024, 2048, 1024, 512]).cuda()


        self.agent = A3cAgent(self.actiom_num,self.lnet,0)
        self.agent01 = A3cAgent(self.actiom_num,self.lnet01,1)
        self.agent02 = A3cAgent(self.actiom_num,self.lnet02,2)

        self.pop = torch.optim.Adam(self.lnet.parameters(), lr=0.0001)
        self.pop01 = torch.optim.Adam(self.lnet01.parameters(), lr=0.0001)
        self.pop02 = torch.optim.Adam(self.lnet02.parameters(), lr=0.0001)

        self.max_episode = max_episode
        self.update_global_iteration = update_global_iteration
        self.gamma = gamma
        self.normalizer = Normalizer()
        self.loss = 0
        self.rr = 0
        self.rr01 = 0
        self.rr02 = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def progress_bar(self,portion, total):

        part = total / 50  # 1%数据的大小
        count = math.ceil(portion / part)
        sys.stdout.write('\r')
        sys.stdout.write(('[%-50s]%.2f%%' % (('>' * count), portion / total * 100)))
        sys.stdout.flush()

        if portion >= total:
            sys.stdout.write('\n')
            return True


    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def train(self):
        normal_size = 0
        episode = 0
        sum_reward = [0,0,0]
        while episode < self.max_episode:

            #定义程序开始时间
            m_s = datetime.datetime.now()
            #定义程序运行环境
            self.agent.updateMode(self.lnet)
            self.agent01.updateMode(self.lnet01)
            self.agent02.updateMode(self.lnet02)

            rhcp_shang_agent.num = 0

            self.agent.num = 0
            self.agent01.num = 0
            self.agent02.num = 0
            self.env.set_agents([self.agent,self.agent01,self.agent02])
            trajectories,pay_off = self.env.run(is_training=True)

            buffer_s, buffer_a, buffer_r,buffer_prob,buffer_s_,buffer_done = [], [], [],[],[],[]
            buffer_s01, buffer_a01, buffer_r01,buffer_prob01,buffer_s01_,buffer_done01 = [], [], [],[],[],[]
            buffer_s02, buffer_a02, buffer_r02,buffer_prob02,buffer_s02_,buffer_done02 = [], [], [],[],[],[]


            #填充正则化项
            if normal_size < 1000:
                print("第",episode,"轮==添加数据池===============",normal_size,pay_off)
                for ts in trajectories[0]:
                    buffer_s.append(ts[0]["obs"])
                    buffer_a.append(ts[1])
                    buffer_r.append(ts[2])
                    buffer_s_.append(ts[3]["obs"])
                    buffer_done.append(1-ts[2])

                buffer_prob = self.agent.agent_probs
                done = pay_off[0]

                for ts in trajectories[1]:
                    buffer_s01.append(ts[0]["obs"])
                    buffer_a01.append(ts[1])
                    buffer_r01.append(ts[2])
                    buffer_s01_.append(ts[3]["obs"])
                    buffer_done01.append(1-ts[2])

                buffer_prob01 = self.agent01.agent_probs
                done01 = pay_off[1]

                for ts in trajectories[2]:
                    buffer_s02.append(ts[0]["obs"])
                    buffer_a02.append(ts[1])
                    buffer_r02.append(ts[2])
                    buffer_s02_.append(ts[3]["obs"])
                    buffer_done02.append(1-ts[2])

                buffer_prob02 = self.agent02.agent_probs
                done02 = pay_off[2]

                if done01 and done02:

                    if len(trajectories[1]) > len(trajectories[2]):
                        pay_off[2] = 0
                    else:
                        pay_off[1] = 0

                sum_reward = sum_reward + pay_off

                if done:
                    v_s_ = 1
                else:

                    v_s_ = 0.001

                if done01:
                    v_s_01 = 1  # terminal
                else:

                    v_s_01 = 0.001

                if done02:
                    v_s_02 = 1
                else:

                    v_s_02 = 0.001

                buffer_v_target = []
                buffer_v_target01 = []
                buffer_v_target02 = []

                for r,d in zip(buffer_r[::-1],buffer_done[::-1]):
                    v_s_ = r + self.gamma * v_s_ * d
                    buffer_v_target.append(v_s_)

                for r,d in zip(buffer_r01[::-1],buffer_done01[::-1]):
                    v_s_01= r + self.gamma * v_s_01 * d
                    buffer_v_target01.append(v_s_01)

                for r,d in zip(buffer_r02[::-1],buffer_done02[::-1]):
                    v_s_02 = r + self.gamma * v_s_02 * d
                    buffer_v_target02.append(v_s_02)

                buffer_v_target.reverse()
                buffer_v_target01.reverse()
                buffer_v_target02.reverse()

                for i in range(len(buffer_v_target)):
                    memdic = {}
                    memdic['obs'] = buffer_s[i]
                    memdic['action'] = buffer_a[i]
                    memdic['val'] = buffer_v_target[i]
                    memdic['prob'] = buffer_prob[i]
                    self.rmem.addmemory(memdic)

                for i in range(len(buffer_v_target01)):
                    memdic = {}
                    memdic['obs'] = buffer_s01[i]
                    memdic['action'] = buffer_a01[i]
                    memdic['val'] = buffer_v_target01[i]
                    memdic['prob'] = buffer_prob01[i]
                    self.rmem01.addmemory(memdic)

                for i in range(len(buffer_v_target02)):
                    memdic = {}
                    memdic['obs'] = buffer_s02[i]
                    memdic['action'] = buffer_a02[i]
                    memdic['val'] = buffer_v_target02[i]
                    memdic['prob'] = buffer_prob02[i]
                    self.rmem02.addmemory(memdic)
                normal_size = normal_size + 2
            else:
                for ts in trajectories[0]:
                    buffer_s.append(ts[0]["obs"])
                    buffer_a.append(ts[1])
                    buffer_r.append(ts[2])
                    buffer_s_.append(ts[3]["obs"])
                    buffer_done.append(1 - ts[2])

                buffer_prob = self.agent.agent_probs
                done = pay_off[0]

                for ts in trajectories[1]:
                    buffer_s01.append(ts[0]["obs"])
                    buffer_a01.append(ts[1])
                    buffer_r01.append(ts[2])
                    buffer_s01_.append(ts[3]["obs"])
                    buffer_done01.append(1 - ts[2])

                buffer_prob01 = self.agent01.agent_probs
                done01 = pay_off[1]

                for ts in trajectories[2]:
                    buffer_s02.append(ts[0]["obs"])
                    buffer_a02.append(ts[1])
                    buffer_r02.append(ts[2])
                    buffer_s02_.append(ts[3]["obs"])
                    buffer_done02.append(1 - ts[2])

                buffer_prob02 = self.agent02.agent_probs
                done02 = pay_off[2]

                if done01 and done02:
                    if len(trajectories[1]) > len(trajectories[2]):
                        pay_off[2] = 0
                    else:
                        pay_off[1] = 0

                sum_reward = sum_reward + pay_off

                if done:
                    v_s_ = 1
                else:

                    v_s_ = 0.001

                if done01:
                    v_s_01 = 1  # terminal
                else:
                    v_s_01 = 0.001

                if done02:
                    v_s_02 = 1
                else:
                    v_s_02 = 0.001


                buffer_v_target = []
                buffer_v_target01 = []
                buffer_v_target02 = []

                for r,d in zip(buffer_r[::-1],buffer_done[::-1]):  # reverse buffer r

                    v_s_ = r + self.gamma * v_s_ * d
                    buffer_v_target.append(v_s_)

                for r,d in zip(buffer_r01[::-1],buffer_done01[::-1]):  # reverse buffer r

                    v_s_01= r + self.gamma * v_s_01 * d
                    buffer_v_target01.append(v_s_01)

                for r,d in zip(buffer_r02[::-1],buffer_done02[::-1]):  # reverse buffer r

                    v_s_02 = r + self.gamma * v_s_02 * d
                    buffer_v_target02.append(v_s_02)

                buffer_v_target.reverse()
                buffer_v_target01.reverse()
                buffer_v_target02.reverse()

                for i in range(len(buffer_v_target)):
                    memdic = {}
                    memdic['obs'] = buffer_s[i]
                    memdic['action'] = buffer_a[i]
                    memdic['val'] = buffer_v_target[i]
                    memdic['prob'] = buffer_prob[i]
                    self.rmem.addmemory(memdic)

                for i in range(len(buffer_v_target01)):
                    memdic = {}
                    memdic['obs'] = buffer_s01[i]
                    memdic['action'] = buffer_a01[i]
                    memdic['val'] = buffer_v_target01[i]
                    memdic['prob'] = buffer_prob01[i]
                    self.rmem01.addmemory(memdic)

                for i in range(len(buffer_v_target02)):
                    memdic = {}
                    memdic['obs'] = buffer_s02[i]
                    memdic['action'] = buffer_a02[i]
                    memdic['val'] = buffer_v_target02[i]
                    memdic['prob'] = buffer_prob02[i]
                    self.rmem02.addmemory(memdic)

                bs, ba, bv, bp = self.rmem.sample(64)
                bs = self.v_wrap(np.array(bs))
                ba = self.v_wrap(np.array(ba))
                bv = self.v_wrap(np.array(bv))
                bp = self.v_wrap(np.array(bp))

                self.loss = self.lnet.loss_func(bs, ba, bv, bp)
                self.pop.zero_grad()
                self.loss.backward()
                self.pop.step()

                bs01, ba01, bv01, bp01 = self.rmem01.sample(32)
                bs01 = self.v_wrap(np.array(bs01))
                ba01 = self.v_wrap(np.array(ba01))
                bv01 = self.v_wrap(np.array(bv01))
                bp01 = self.v_wrap(np.array(bp01))

                loss01 = self.lnet01.loss_func(bs01, ba01, bv01, bp01)
                self.pop01.zero_grad()
                loss01.backward()
                self.pop01.step()

                bs02, ba02, bv02, bp02 = self.rmem02.sample(32)
                bs02 = self.v_wrap(np.array(bs02))
                ba02 = self.v_wrap(np.array(ba02))
                bv02 = self.v_wrap(np.array(bv02))
                bp02 = self.v_wrap(np.array(bp02))

                loss02 = self.lnet02.loss_func(bs02, ba02, bv02, bp02)
                self.pop02.zero_grad()
                loss02.backward()
                self.pop02.step()
                """
                if len(self.agent.mem.memory) > 0:
                    self.agent.train_sl()

                if len(self.agent01.mem.memory) > 0:
                    self.agent01.train_sl()

                if len(self.agent02.mem.memory) > 0:
                    self.agent02.train_sl()
                """

            reward = 0
            reward01 = 0
            reward02 = 0

            if episode % eval_every_eposide == 0:

                self.lnet.remove_noise()
                self.lnet01.remove_noise()
                self.lnet02.remove_noise()

                self.agent.updateMode(self.lnet)
                self.agent01.updateMode(self.lnet01)
                self.agent02.updateMode(self.lnet02)

                self.agent.num = 0
                self.eval_env.set_agents([self.agent, rhcp_shang_agent, rhcp_shang_agent01])

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

                logger.add_point(x=episode / eval_every_eposide, y=float(reward) / evaluate_num,z=float(reward01) / evaluate_num, m=float(reward02) / evaluate_num)
                self.res_queue.put(self.rr)

                sum_reward = [0, 0, 0]

                self.lnet.sample_noise(self.rr,0)
                self.lnet01.sample_noise(self.rr01+self.rr01,1)
                self.lnet02.sample_noise(self.rr01+self.rr01,2)

            # if episode % save_plot_every == 0 and episode > 0:
            # logger.make_plot(save_path=path + '/' + self.name + "-" + str(episode) + '.png')s

            if episode % clear_memory_every_eposide == 0 :
                self.rmem.clear()
                self.rmem01.clear()
                self.rmem02.clear()
                normal_size = 0

            if episode % save_mode_every_eposide == 0 and episode > 0:
                torch.save(self.lnet, path01 + "/" + self.name + "nework.pkl")
                torch.save(self.lnet01, path01 + "/" + self.name + "nework01.pkl")
                torch.save(self.lnet02, path01 + "/" + self.name + "nework02.pkl")

                torch.save(self.agent.snet, path01 + "/" + self.name + "snet" + str(self.agent.id) + ".pkl")
                torch.save(self.agent01.snet, path01 + "/" + self.name + "snet" + str(self.agent01.id) + ".pkl")
                torch.save(self.agent02.snet, path01 + "/" + self.name + "snet" + str(self.agent02.id) + ".pkl")

                if episode % (save_mode_every_eposide*10) == 0:
                    torch.save(self.lnet, path01 + "/" +str(episode)+self.name + "nework.pkl")
                    torch.save(self.lnet01, path01 + "/" +str(episode)+ self.name + "nework01.pkl")
                    torch.save(self.lnet02, path01 + "/" +str(episode)+ self.name + "nework02.pkl")
                print("****************保存模型----", path01 + "/" + self.name + "nework.pkl*************")

            m_e = datetime.datetime.now()
            sec = m_e - m_s
            print("\n", "worker:", self.name, "the epsoide is:", episode, "loss is:", 0, "cost is:", sec.seconds,"single_reward:", pay_off, "sum_reward:", sum_reward, "reward is:", self.rr, "::", self.rr01, "::",self.rr02)
            episode = episode + 1
