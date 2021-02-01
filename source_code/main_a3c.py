#coding=utf-8
import rlcard
from game_model.shared_adam import SharedAdam
from net_work.a3c_neural_network import Net
import torch.multiprocessing as mp
from game_model.a3c_doudizhu_game_agent import Worker
import threading
import datetime
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    m_s = datetime.datetime.now()
    env = rlcard.make('doudizhu')
    state_shape = env.state_shape
    actiom_num = env.action_num
    net_layers = [512,1024,2048,1024,512]

    gnet = Net(actiom_num,state_shape,net_layers) # global network
    print("gnet is:",gnet)
    gnet.share_memory()  # share the global parameters in multiprocessing

    opt = SharedAdam(gnet.parameters(), lr=0.00005)  # global optimizer

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.),mp.Queue()

    # parallel training

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue,i,400000,10,0.9) for i in range(1)]

    threads = [threading.Thread(target=worker.train) for worker in workers]

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m_e = datetime.datetime.now()

    sec = m_e - m_s
    print("执行完成,耗时",sec.seconds)
