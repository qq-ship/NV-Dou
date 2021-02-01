#coding=utf-8
import pandas as pd
import numpy as np
import scipy.interpolate, scipy.optimize
import matplotlib.pyplot as plt

EV_Zong = 10
def ReadData(path):
    data = pd.read_csv(path, header=None)
    dataArray = np.array(data.values)
    x = []
    y = []
    for i in range(len(dataArray)):
        x.append(dataArray[i][0])
        y.append(dataArray[i][1])
    return x,y

if __name__ == "__main__":

    x, y = ReadData("result_data/pll.csv") #传统DQN

    """
    ty = []
    for i in range(len(y)):
        if i % 2 == 0:
            ty.append(y[i])
    tx = [i for i in range(len(ty))]

    x = tx
    y = ty
    print(np.array(x).shape)
    print(np.array(y).shape)
    """



    ll = []
    sum = 0.0

    zong = []

    kk = 0
    for i in range(len(y)):
        kk = kk + y[i]
        if (i+1) % EV_Zong == 0:
            kk = kk/EV_Zong
            zong.append(kk)
            kk = 0

    print(len(zong))

    zongx = np.arange(len(zong))+1
    #print(zongx)
    #print(zong)
    zongx = zongx * EV_Zong
    """
    x1,y1 = ReadData("result_data/lstm-dqn-rs.csv")#LSTM DQN
    x2,y2 =  ReadData("result_data/20191225-fu-01.csv")#LSTM TQ DQN TQ3
    xt,yt = ReadData("result_data/20191225rl.csv")#LSTM TQ DQN TQ3
    """
    """
    rw = sum(y)*100
    rw1 = sum(y1)*100
    rw2 = sum(yt)*100
    """
    #print("rl_dqn:",rw,"lstm_dqn:",rw1,"tq(lmda):",rw2)
    plt.plot(x,y, marker='o', mec='none', ms=3, lw=1, label='y1')
    plt.plot(zongx,zong, marker='o', mec='none', ms=3, lw=2, label='y1')
    #plt.plot(x,y, marker='o', mec='none',ms=3, lw=1, label='y1')
    #plt.plot(x1, y1, marker='o', mec='none',ms=3, lw=1, label='y2')
    #plt.plot(x2, y2, marker='o', mec='none', ms=3, lw=1, label='y3')
    #plt.legend(['rl_dqn','lstm_dqn'])  #
    plt.grid()
    plt.show()
