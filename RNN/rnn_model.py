from torch import nn

class WeldRNN(nn.Module):
    def __init__(self,
                 input_size = 6,
                 hidden_size = 1024,
                 output_size = 1,
                 num_layers = 1,
                 dropout = 0
                 ):
        super().__init__()
        self.rnn=nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          nonlinearity='relu',
                          dropout = dropout
                          )
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

    def forward(self,src):
        output, state = self.rnn(src)
        output = self.linear(output)

        return output
