import rlcard
import datetime
import os
import torch
import numpy as np
from op_model.rhcp_shang_agent import RhcpShangAgent
import random
from net_work.abstract_neural_network import ANet

update_ep = 1
save_mode_every = 100

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
        if len(self.memory) <=  num:
            rs = self.memory
            for i in range(len(rs)):
                obs.append(rs[i]['obs'])
                act.append(rs[i]['action'])

            return obs,act
        else:
            rs = random.sample(self.memory, num)
            for i in range(len(rs)):
                obs.append(rs[i]['obs'])
                act.append(rs[i]['action'])
            return obs,act

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class A3cAgent(object):

    def __init__(self,action_num,lnet):

        self.action_num = action_num
        self.lnet = lnet

    def updateMode(self,newlnet):
        self.lnet = newlnet

    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)


    def step(self,state,player_id):
        s = np.array(state['obs'])

        action = self.lnet.choose_action(np.expand_dims(s, 0),state["legal_actions"])

        return int(action)

    def eval_step(self,state,player_id):

        return self.step(state,player_id)


def train(eposide):
    mem = Memory(10000)
    anet = ANet(309,[6,5,15], [512, 1024, 2048, 1024, 512])
    agent = A3cAgent(309,anet)
    op = torch.optim.Adam(anet.parameters(), lr=0.0001)
    rhcp_shang_agent = RhcpShangAgent(309)
    nowTime = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    path = "out_put/anets/a3c/" + nowTime
    os.makedirs(path)
    reward = 0

    env = rlcard.make('doudizhu')

    ep = 0

    while ep < eposide:
        rhcp_shang_agent.num = 0
        env.set_agents([rhcp_shang_agent, rhcp_shang_agent, rhcp_shang_agent])
        trajectories01, pay_off01 = env.run(is_training=True)

        for ts in trajectories01[0]:
            memdic = {}
            memdic['obs'] = ts[0]["obs"]
            memdic['action'] = ts[1]
            mem.addmemory(memdic)

        if ep % update_ep == 0:
            bbs, bba = mem.sample(100)
            bbs = v_wrap(np.array(bbs))
            bba = v_wrap(np.array(bba))
            bba = bba.long()

            bloss,tong = anet.loss_func(bbs, bba)
            op.zero_grad()
            bloss.backward()
            op.step()
        else:
            bloss = 0
            tong = 0


        if ep % save_mode_every == 0:
            print("模型保存")
            torch.save(anet,path + "/" + "anework.pkl")

        if ep % 10 == 0 and ep !=0:
            agent.updateMode(anet)
            env.set_agents([agent, rhcp_shang_agent,rhcp_shang_agent])
            for i in range(100):
                # self.progress_bar(i+1,500)
                _, pay_off = env.run(is_training=False)
                reward = reward + pay_off[0]
                print(pay_off)

        print("the ep is:",ep,"the bloss is:",bloss,"the tong is:",tong,"reward:",reward)


        ep = ep + 1

if __name__ == "__main__":
    train(30000)