import sys
import torch
import numpy as np
from torch.linalg import pinv
from torch import tanh, sigmoid, diag, square
import matplotlib.pyplot as plt

# load internal packages
from linearization import lstm_linearization, tanh_p, sigmoid_p
sys.path.append("../multi_output/")
from lstm_model_next_step import WeldLSTM
from lstm_train_next_step import WeldDataset


if __name__=='__main__':
    MAX_IDX = 50
    HID_DIM = 8
    HEIGHT_REF = 1.3
    i = 0
    # load data for mean
    TRAIN_DATA_DIR = '../data/lstm_processed/CL_cold.npy'
    train_dataset = WeldDataset(TRAIN_DATA_DIR)

    setpoint = ((torch.ones(MAX_IDX)*HEIGHT_REF)-train_dataset.mean[3])/train_dataset.std[3]

    # load model
    model = torch.load('../multi_output/saved_model_8_next_step.pt')
    model.eval()

    # initialize hidden state
    h = torch.zeros(1,HID_DIM)
    c = torch.zeros(1,HID_DIM)
    state = (h,c)
    u_prev = 0.0
    T_prev = 0.0
    dh_prev = 0.0

    u_cmds = []
    dh = []
    dh_est = []
    idxs = np.linspace(1,MAX_IDX,MAX_IDX)
    print(idxs)
    
    while i<MAX_IDX:
        # calculate linearization
        h_0 = torch.squeeze(state[0])
        c_0 = torch.squeeze(state[1])
        u_0 = torch.tensor([u_prev, T_prev, dh_prev])

        y_0,_ = model(torch.unsqueeze(u_0, dim=0), 
                            hidden_state=state)

        y_0 = torch.squeeze(y_0)

        A,B,C = lstm_linearization(model, h_0, c_0, u_0)

        # isolate the effect of the velocity on the height input
        B = B[:,0]
        C = C[1,:]

        # generate velocity profile according to optimization
        y_d = torch.unsqueeze(setpoint[i], dim=0)

        u_cmd = y_d/(C@B)
        x = torch.unsqueeze(torch.tensor([u_cmd, T_prev, dh_prev]),dim=0)
        y_out, state = model(x, hidden_state=state)

        # update prev variables
        u_prev = u_cmd
        T_prev = torch.squeeze(y_out)[0]
        dh_prev = torch.squeeze(y_out)[1]

        # save relevant outputs
        u_cmds.append(u_cmd.detach())
        dh.append(dh_prev.detach())
        dh_est.append(((C@B)*u_cmd).detach())

        i+=1
    
    u_cmds = np.array(u_cmds)
    dh = np.array(dh)
    dh_est = np.array(dh_est)
    u_cmds = u_cmds*train_dataset.std[0]+train_dataset.mean[0]
    dh = dh*train_dataset.std[3]+train_dataset.mean[3]
    dh_est = dh_est*train_dataset.std[3]+train_dataset.mean[3]

    fig,ax = plt.subplots(2,1, sharex = True)
    ax[0].plot(dh)
    ax[0].scatter(idxs, dh_est)
    ax[1].plot(u_cmds)
    plt.show()



        
