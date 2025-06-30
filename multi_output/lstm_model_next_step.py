from torch import nn
import torch

class WeldLSTM(nn.Module):
    def __init__(self,
                 input_size = 6,
                 hidden_size = 1024,
                 output_size = 1,
                 num_layers = 1,
                 dropout = 0
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout = dropout
                          )
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

    def forward(self,src):
        if self.training:
            output, state = self.lstm(src)
            output = self.linear(output)

        else:
            # should be testing
            # one batch at a time
            # loop through sequence
            output = torch.zeros((src.shape[-2], self.output_size))
            # print("Working Output")
            # print(output.shape)
            state = None
            for idx, val in enumerate(src[0]):
                val_temp = val.unsqueeze(0)
                if state is None:
                    out, state = self.lstm(val_temp)
                    output[idx,:] = self.linear(out)
                else:
                    # construct input from previous output
                    _input = torch.Tensor([[val_temp[:,0], output[idx-1,0], output[idx-1,0]]])
                    out, state = self.lstm(_input, state)
                    output[idx,:] = self.linear(out)

        return output

            

