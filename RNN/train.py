import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import WeldLSTM

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model parameters
    INPUT_SIZE = 7
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    NUM_LAYERS = 1

    # hyper parameters
    BATCH_SIZE = 20
    MAX_EPOCH = 2000
    LR = 0.005
    WD = 0.0001

    # model
    model = WeldLSTM(INPUT_SIZE,
                     HIDDEN_SIZE,
                     OUTPUT_SIZE,
                     NUM_LAYERS)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=WD)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                              lr=LR)
    # loss
    loss_fn = nn.MSELoss()

    # load data
    TRAIN_DATA_DIR = '../data/processed/CL_cold.npy'
    VALID_DATA_DIR = '../data/processed/CL_hot.npy'

    train_dataset = WeldDataset(TRAIN_DATA_DIR)
    valid_dataset = WeldDataset(VALID_DATA_DIR)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle = True
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle = False
                                  )
    
    # initialize storage lists
    train_losses = []
    valid_losses = []
    best_loss = 1e6

    # Training Loop
    for epoch in range(MAX_EPOCH):
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
        for src,trg in tqdm(valid_dataloader):
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

        print()
        print(f'-----Epoch {epoch} Stats-----')
        print(f'Training Loss: {train_loss}')
        print(f'Valid Loss:    {valid_loss}')

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
    def __init__(self,filepath):
        # TODO: Deal with nan in data
        data = np.load(filepath)
        self.trg = torch.Tensor(data[:,:,-1])
        self.src = torch.Tensor(data[:,:,:-1])
        
        # fliter nan values
        self.trg = torch.nan_to_num(self.trg)
        self.src = torch.nan_to_num(self.src)

        # normalize
        self.src = nn.functional.normalize(self.src)

    def __len__(self):
        return self.trg.shape[0]

    def __getitem__(self, idx):
        return self.src[idx,:,:], self.trg[idx,:]

if __name__== '__main__':
    train()
