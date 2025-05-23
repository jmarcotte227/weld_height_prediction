import sys
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import WeldLSTM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def rmse(e):
    sum = 0
    for err in e:
        sum += err**2

    return np.sqrt(sum/len(e))

def test():
    # load model
    model = torch.load('saved_model.pt')
    # load data
    VALID_DATA_DIR = '../data/processed/CL_hot.npy'

    valid_dataset = WeldDataset(VALID_DATA_DIR)
    nonorm_dataset = WeldDataset(VALID_DATA_DIR, norm=False)

    # initialize log-log model
    llmodel = SpeedHeightModel(a=-0.36997977, b=1.21532975)

    errors = []
    errors_ll = []
    for idx, (src,trg) in enumerate(valid_dataset):
        # lstm error
        pred_lstm = model(src)
        pred_lstm = pred_lstm.squeeze()
        error = trg-pred_lstm
        print(error)

        errors = errors + error.tolist()

        # static model error
        src_nonorm, _ = nonorm_dataset[idx] 
        vel = src_nonorm[:,1]
        pred_ll = llmodel.v2dh(vel)

        error_ll = trg-pred_ll

        errors_ll = errors_ll + error_ll.tolist()

        print(f'lstm error: {sum(error)/len(error)}')
        print(f'll error:   {sum(error_ll)/len(error_ll)}')

    # print('-----------------')
    # print(f'LSTM RMSE:    {rmse(errors)}')
    # print(f'Log-Log RMSE: {rmse(errors_ll)}')



class WeldDataset(Dataset):
    '''
    To be used with the processed files from a welding run.
    '''
    def __init__(self,filepath,norm=True):

        data = np.load(filepath)
        print(data.shape)
        self.trg = torch.Tensor(data[:,:,-1])
        self.src = torch.Tensor(data[:,:,:-1])
        
        # fliter nan values
        self.trg = torch.nan_to_num(self.trg)
        self.src = torch.nan_to_num(self.src)

        # normalize
        if norm: 
            self.src = nn.functional.normalize(self.src)

    def __len__(self):
        return self.trg.shape[0]

    def __getitem__(self, idx):
        return self.src[idx,:,:], self.trg[idx,:]
class SpeedHeightModel:
    """
    Model relating dh to torch speed according to the equation
    ln(h) = a ln(v) + b
    """

    def __init__(self, lam=0.05, beta=0.99, a=-0.4619, b=1.647, p = None):
        # Beta == 1 for non-exponentail updates
        self.coeff_mat = np.array([a, b])
        self.nom_a = a
        self.nom_b = b
        self.lam = lam
        if p is None:
            self.p = np.diag(np.ones(self.coeff_mat.shape[0]) * self.lam)
        else: self.p = p
        self.beta = beta

    def v2dh(self, v):
        """outputs the height for a velocity or array of velocities"""
        logdh = self.coeff_mat[0] * np.log(v) + self.coeff_mat[1]

        dh = np.exp(logdh)
        return dh

    def dh2v(self, dh):
        """outputs the velocity for a height or set of heights"""
        logdh = np.log(dh)
        logv = (logdh - self.coeff_mat[1]) / self.coeff_mat[0]

        v = np.exp(logv)
        return v

    def model_update_rls(self, vels, dhs):
        """updates the model coefficients using the recursive
        least-squares algorithm"""
        # Algorithm from
        # https://osquant.com/papers/recursive-least-squares-linear-regression/
        vels = np.reshape(vels,-1)
        for idx, vel in enumerate(vels):
            x = np.array([[np.log(np.array(vel))], [1]])
            y = np.log(dhs[idx])
            if not np.isnan(y):
                r = 1 + (x.T @ self.p @ x) / self.beta
                k = self.p @ x / (r * self.beta)
                e = y - x.T @ self.coeff_mat
                self.coeff_mat = self.coeff_mat + k @ e
                self.p = self.p / self.beta - k @ k.T * r

if __name__=='__main__':
    test()
