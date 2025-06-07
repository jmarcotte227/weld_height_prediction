import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize, Bounds
from numpy.linalg import norm

sys.path.append('../LSTM/')
import lstm_model 
import lstm_train

# optimization function
def v_opt(v_T, dh_targ, model):
    v_T = torch.unsqueeze(torch.tensor(v_T, dtype=torch.float32), dim=1)
    h_dif = norm(dh_targ - model(v_T).detach().numpy()) ** 2
    print(h_dif)
    return h_dif

def standardize(x, mean, std):
    return (x-mean)/std
    
if __name__=='__main__':


    # constants
    VEC_LEN =   45
    v_min =     3
    v_max =     17

    dh_nom =    0.5

    # load model
    lstm = torch.load('../LSTM/saved_model_vset.pt')
    lstm.requires_grad_(False)

    # load mean and std
    mean = torch.load('../LSTM/mean.pt')[0]
    std = torch.load('../LSTM/std.pt')[0]
    v_min = standardize(v_min,mean,std)
    print(v_min)
    v_max = standardize(v_max,mean,std)
    print(v_max)
    # print(mean)
    # print(std)

    # setup optimization
    # bounds = Bounds(V_MIN, V_MAX)
    v_T = 5*torch.ones(VEC_LEN, dtype=torch.float32)
    v_T = standardize(v_T, mean, std)

    # target deposition
    dh_des = torch.unsqueeze(torch.tensor(dh_nom*np.ones(45), dtype=torch.float32), dim=1)

    # setup input as parameter with grad
    v_T = torch.nn.Parameter(torch.unsqueeze(v_T, dim=1), requires_grad=True)
    optim = torch.optim.SGD([v_T], lr=5e-1)
    mse = torch.nn.MSELoss()


    num_steps = 5000

    for _ in range(num_steps):
        res = lstm(v_T)
        loss = mse(res, dh_des)
        loss.backward()
        optim.step()
        v_T.data.clamp_(v_min, v_max)
        
        optim.zero_grad()
        # plt.plot(res.detach())
        # plt.show()

    dh_pred = lstm(v_T)
    vel_out = v_T*std+mean
    # print(v_T*std+mean)

    # ploting result
    fig,ax = plt.subplots(2,1)

    ax[0].plot(dh_des.detach())
    ax[0].plot(dh_pred.detach())
    ax[0].set_ylim([0,3])
    ax[1].plot(vel_out.detach())
    plt.show()
