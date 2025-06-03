import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch


# import models
sys.path.append('RNN/')
import rnn_model 
import rnn_train
sys.path.append('LSTM/')
import lstm_model 
import lstm_train

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

def run_model(model, valid_dataset):
    trgs = []
    preds = []
    times = []
    for idx, (src,trg) in enumerate(valid_dataset):
        # timing single sequence
        start_time = time.time()
        pred = model(src)
        end_time = time.time()
        times.append(end_time-start_time)
        pred = pred.squeeze()
        preds.append(pred.tolist())
        trgs.append(trg.tolist())
    return trgs, preds, times

def run_log_log(model, valid_dataset):
    '''
    Use non-standardized data for log-log model
    '''
    trgs = []
    preds = []
    times = []
    for idx, (src, trg) in enumerate(valid_dataset):
        vel = src[:,0]
        start_time = time.time()
        pred = model.v2dh(vel)
        end_time = time.time()
        times.append(end_time-start_time)
        preds.append(pred.tolist())
        trgs.append(trg.tolist())
        
    return trgs, preds, times

def rmse(e):
    sum = 0
    for err in e:
        sum += err**2

    return np.sqrt(sum/len(e))

def mae(e):
    sum = 0
    for err in e:
        sum+=np.abs(err)
    return sum/len(e)

def calc_error(trgs, preds):
    pred_set = []
    trg_set = []

    # concatenate all target and prediction
    for idx, _ in enumerate(preds):
        pred_set = pred_set + preds[idx]
        trg_set = trg_set + trgs[idx]

    error = [x-y for x,y in zip(pred_set,trg_set)]
    return error

if __name__=='__main__':

    LSTM_FLAG = True
    RNN_FLAG = True
    LL_FLAG = True

    # data dir filepaths
    LSTM_DATA_DIR = 'data/lstm_processed/CL_hot.npy'
    RNN_DATA_DIR = 'data/RNN_processed/CL_hot.npy'
    NONORM_DATA_DIR = 'data/RNN_processed/CL_hot.npy' # arbitrarily chosen, could use either one

    # load mean and std for LSTM and RNN
    lstm_mean = torch.load('LSTM/mean.pt')
    lstm_std = torch.load('LSTM/std.pt')
    rnn_mean = torch.load('RNN/mean.pt')
    rnn_std = torch.load('RNN/std.pt')

    # initialize datasets
    if LSTM_FLAG: lstm_dataset = lstm_train.WeldDataset(LSTM_DATA_DIR, lstm_mean, lstm_std)
    if RNN_FLAG: rnn_dataset = rnn_train.WeldDataset(RNN_DATA_DIR, lstm_mean, lstm_std)
    if LL_FLAG: loglog_dataset = rnn_train.WeldDataset(NONORM_DATA_DIR, norm=False)

    # load models
    lstm = torch.load('LSTM/saved_model.pt')
    rnn = torch.load('RNN/saved_model.pt')
    loglog = SpeedHeightModel(a=-0.36997977, b=1.21532975)

    # test models
    if LSTM_FLAG: lstm_trg, lstm_pred, lstm_times = run_model(lstm, lstm_dataset)
    if RNN_FLAG: rnn_trg, rnn_pred, rnn_times = run_model(rnn, rnn_dataset)
    if LL_FLAG: ll_trg, ll_pred, ll_times = run_log_log(loglog, loglog_dataset)

    # plot example layer
    while True:
        try:
            layer = int(input("Enter Layer: "))
            fig,ax = plt.subplots(1,1)
            ax.plot(ll_trg[layer], label = "Measured Height")
            if LL_FLAG: ax.plot(ll_pred[layer], label = "LL")
            if RNN_FLAG: ax.plot(rnn_pred[layer], label = "RNN")
            if LSTM_FLAG: ax.plot(lstm_pred[layer], label = "LSTM")
            ax.set_title(f"Layer {layer}")
            ax.set_xlabel("Segment Index")
            ax.set_ylabel("$\Delta h$")
            ax.legend()
            plt.show()
        except KeyboardInterrupt:
            break

    # calculate errors
    if LSTM_FLAG: 
        lstm_err = calc_error(lstm_trg, lstm_pred)
        print(f"LSTM RMSE: {rmse(lstm_err):.3f}")
        print(f"LSTM MAE:  {mae(lstm_err):.3f}")
        print(f"LSTM Max:  {max(lstm_err):.3f}")
        print(f"LSTM Time: {sum(lstm_times)/len(lstm_times):.6f}")
        print("----------------------------")
    if RNN_FLAG: 
        rnn_err = calc_error(rnn_trg, rnn_pred)
        print(f"RNN RMSE:  {rmse(rnn_err):.3f}")
        print(f"RNN MAE:   {mae(rnn_err):.3f}")
        print(f"RNN Max:   {max(rnn_err):.3f}")
        print(f"RNN Time:  {sum(rnn_times)/len(rnn_times):.6f}")
        print("----------------------------")
    if LL_FLAG: 
        ll_err = calc_error(ll_trg, ll_pred)
        print(f"LL RMSE:   {rmse(ll_err):.3f}")
        print(f"LL MAE:    {mae(ll_err):.3f}")
        print(f"LL Max:    {max(ll_err):.3f}")
        print(f"LL Time:   {sum(ll_times)/len(ll_times):.6f}")

