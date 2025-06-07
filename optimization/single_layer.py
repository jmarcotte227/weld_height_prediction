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
    return (
        norm(dh_targ - model(v_T).detach().numpy()) ** 2
    )

def to_tensor(x):
    return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), dim=1)

if __name__=='__main__':
    # constants
    VEC_LEN =   45
    V_MIN =     0
    V_MAX = 17     

    # load model
    lstm = torch.load('../LSTM/saved_model_vset.pt')

    # setup optimization
    bounds = Bounds(V_MIN, V_MAX)
    v_init = 0.5*np.ones(VEC_LEN)

    # target deposition
    dh_des = 2.0*np.ones(45)

    # perform minimization
    opt_result = minimize(
        v_opt,
        v_init,
        (dh_des, lstm),
        # bounds=bounds,
        options={"maxfun": 100000},
    )

    v_opt = opt_result.x
    print(opt_result)

    # convert to tensor

    fig,ax = plt.subplots(1,1)
    ax.plot(lstm(to_tensor(v_opt)).detach())
    plt.show()
