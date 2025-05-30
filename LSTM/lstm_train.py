import sys
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lstm_model import WeldLSTM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train(train_dataset, valid_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model parameters
    INPUT_SIZE = 2
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 1
    NUM_LAYERS = 1

    # hyper parameters
    BATCH_SIZE = 20
    MAX_EPOCH = 1000
    LR = 0.005
    WD = 0.0001
    DROPOUT = 0

    # model
    model = WeldLSTM(INPUT_SIZE,
                     HIDDEN_SIZE,
                     OUTPUT_SIZE,
                     NUM_LAYERS,
                     DROPOUT)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=WD)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                              lr=LR)
    # loss
    loss_fn = nn.MSELoss()


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle = True,
                                  drop_last = True
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle = False
                                  )
    
    # initialize storage lists
    train_losses = []
    valid_losses = []
    best_loss = 1e6

    # setup progress bar
    pbar = tqdm(range(MAX_EPOCH))
    # Training Loop
    for epoch in pbar:
        train_loss = []
        model.train()
        for src,trg in train_dataloader:
            src = src.to(device)
            trg = trg.to(device)
            pred = model(src)

            # squeeze to eliminate dimension of size 1
            pred = torch.squeeze(pred, 2)
            optimizer.zero_grad()
            loss = loss_fn(pred, trg)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # validation
        valid_loss = []
        model.eval()
        for src,trg in valid_dataloader:
            src = src.to(device)
            trg = trg.to(device)

            pred = model(src)

            # squeeze to eliminate dimension of size 1
            pred = torch.squeeze(pred, 2)
            loss = loss_fn(pred, trg)

            valid_loss.append(loss.item())

        # compute averages
        train_loss = sum(train_loss)/len(train_loss)
        valid_loss = sum(valid_loss)/len(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # check if valid loss is better than best and save
        if valid_loss<best_loss:
            torch.save(model, f'saved_model.pt')
            best_loss = valid_loss

        # print()
        # print(f'-----Epoch {epoch} Stats-----')
        # print(f'Training Loss: {train_loss}')
        # print(f'Valid Loss:    {valid_loss}')
        pbar.set_description(f"T/V: {train_loss:.4f}/{valid_loss:.4f}")

    print("----- Final Results -----")
    print(f'Min Valid Loss: {min(valid_losses)}')
    print(f'Epoch: {valid_losses.index(min(valid_losses))}')

    fig,ax = plt.subplots()
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.grid()
    plt.show()

class WeldDataset(Dataset):
    '''
    To be used with the processed files from a welding run.
    '''
    def __init__(self,filepath, mean=None, std=None, norm=True):
        self.mean = mean
        self.std = std
        
        data = np.load(filepath)
        self.trg = torch.Tensor(data[:,:,-1])
        self.src = torch.Tensor(data[:,:,:-1])
        
        # fliter nan values
        self.trg = torch.nan_to_num(self.trg)
        self.src = torch.nan_to_num(self.src)

        # normalize
        temp_data = torch.reshape(self.src, (-1,2))
        if self.mean is None:
            self.mean = temp_data.mean(dim=0)
            self.std = temp_data.std(dim=0)
        if norm:
            for i in range(self.mean.shape[0]):
                self.src[:,:,i] = (self.src[:,:,i]-self.mean[i])/self.std[i]

        # self.src = nn.functional.normalize(self.src)

    def __len__(self):
        return self.trg.shape[0]

    def __getitem__(self, idx):
        return self.src[idx,:,:], self.trg[idx,:]

def rmse(e):
    sum = 0
    for err in e:
        sum += err**2

    return np.sqrt(sum/len(e))
    # return sum/len(e)

def test(valid_dataset, nonorm_dataset):
    # load model
    model = torch.load('saved_model.pt')
    # load data
    VALID_DATA_DIR = '../data/processed/CL_hot.npy'


    # initialize log-log model
    llmodel = SpeedHeightModel(a=-0.36997977, b=1.21532975)
    # llmodel = SpeedHeightModel()

    errors = []
    errors_ll = []
    for idx, (src,trg) in enumerate(valid_dataset):
        # lstm error
        pred_lstm = model(src)
        pred_lstm = pred_lstm.squeeze()
        error = trg-pred_lstm

        errors = errors + error.tolist()

        # static model error
        src_nonorm, _ = nonorm_dataset[idx] 
        vel = src_nonorm[:,0]
        pred_ll = llmodel.v2dh(vel)

        error_ll = trg-pred_ll
        errors_ll = errors_ll + error_ll.tolist()

        # print(f'lstm error: {sum(error)/len(error)}')
        # print(f'll error:   {sum(error_ll)/len(error_ll)}')

    print('-----------------')
    print(f'LSTM RMSE:    {rmse(errors)}')
    print(f'Log-Log RMSE: {rmse(errors_ll)}')

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

if __name__== '__main__':
    # load data
    TRAIN_DATA_DIR = '../data/lstm_processed/CL_cold.npy'
    VALID_DATA_DIR = '../data/lstm_processed/CL_hot.npy'

    train_dataset = WeldDataset(TRAIN_DATA_DIR)
    valid_dataset = WeldDataset(VALID_DATA_DIR, train_dataset.mean, train_dataset.std)
    nonorm_dataset = WeldDataset(VALID_DATA_DIR, norm=False)

    # train model
    train(train_dataset, valid_dataset)
    # test model
    test(valid_dataset, nonorm_dataset)
