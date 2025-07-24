import sys
import torch
from torch import tanh, sigmoid, diag, square

# load internal packages
sys.path.append("../multi_output/")
from lstm_model_next_step import WeldLSTM
from lstm_train_next_step import WeldDataset


def tanh_p(x):
    # use for vector x
    return diag(1-square(tanh(x)))

def sigmoid_p(x):
    return diag(sigmoid(x)*(1-sigmoid(x)))

def lstm_linearization(model, h_0, c_0, u_0):
    # takes the model, hidden state opperating points and input operating
    # point and produces the linearized system matrices. 
    h_dim = h_0.shape[0]
    
    # model weights
    # TODO: needed to add '0' to the end of the variable name. not in the documentation
    W_hi = model.lstm.weight_hh_l0[0*h_dim:1*h_dim,:]
    W_hf = model.lstm.weight_hh_l0[1*h_dim:2*h_dim,:]
    W_hc = model.lstm.weight_hh_l0[2*h_dim:3*h_dim,:]
    W_ho = model.lstm.weight_hh_l0[3*h_dim:4*h_dim,:]

    W_ui = model.lstm.weight_ih_l0[0*h_dim:1*h_dim,:].T
    W_uf = model.lstm.weight_ih_l0[1*h_dim:2*h_dim,:].T
    W_uc = model.lstm.weight_ih_l0[2*h_dim:3*h_dim,:].T
    W_uo = model.lstm.weight_ih_l0[3*h_dim:4*h_dim,:].T

    # model biases
    b = model.lstm.bias_ih_l0+model.lstm.bias_hh_l0
    b_i = b[0*h_dim:1*h_dim]
    b_f = b[1*h_dim:2*h_dim]
    b_c = b[2*h_dim:3*h_dim]
    b_o = b[3*h_dim:4*h_dim]

    ##### Compute B_h #####
    f = sigmoid(W_hf.T@)
    i = sigmoid(W_hi.T@h_0+W_ui.T@u_0+b_i)
    tc = tanh(W_hc.T@h_0+W_uc.T@u_0+b_c)
    c = f*c_0+i*tc

    df_du = W_uf@sigmoid_p(W_hf.T@h_0+W_uf.T@u_0+b_f)
    di_du = W_ui@sigmoid_p(W_hi.T@h_0+W_ui.T@u_0+b_i)
    dtc_du = W_uc@tanh_p(W_hc.T@h_0+W_uc.T@u_0+b_c)

    dc_du = df_du@diag(c_0)+di_du@diag(tc)+dtc_du@diag(i)
    do_du = dc_du@tanh_p(c)
    A_h=0
    B_h=0
    C = 0


    return A_h, B_h, C

if __name__=="__main__":
    # x = torch.Tensor([0,1,2,3])

    # print("Tanh(x): ", tanh(x))
    # print("Tanh_p(x): ", tanh_p(x))
    # print("sigmoid(x): ", sigmoid(x))
    # print("sigmoid_p(x): ", sigmoid_p(x))

    # load data
    TRAIN_DATA_DIR = '../data/lstm_processed/CL_cold.npy'
    VALID_DATA_DIR = '../data/lstm_processed/CL_hot.npy'

    train_dataset = WeldDataset(TRAIN_DATA_DIR)
    valid_dataset = WeldDataset(VALID_DATA_DIR, train_dataset.mean, train_dataset.std)
    nonorm_dataset = WeldDataset(VALID_DATA_DIR, norm=False)

    # load model
    model = torch.load('../multi_output/saved_model_8_next_step.pt')
    model.eval()

    src, trg = valid_dataset[2]

    # warm up model with first 10 steps
    start_seg = 10
    pred_lstm, state = model(torch.unsqueeze(src[:,[0,2,3]], dim=0), 
                             start_seg,
                             stop_seq = True)

    # define operating point
    h_0 = torch.squeeze(state[0])
    c_0 = torch.squeeze(state[1])
    u_0 = torch.squeeze(src[start_seg, [0,2,3]])

    A_h, B_h, C = lstm_linearization(model, h_0, c_0, u_0)





