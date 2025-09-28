from torch import nn
import torch.nn.functional as F

class TST_Decoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, layers, args):  # Fixed the method name
        super(TST_Decoder, self).__init__()  # Fixed the method name
        self.z_dim = inp_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(
            input_size=self.z_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=layers,
            batch_first=True
        )

        self.linear = nn.Linear(self.hidden_dim * 2, args.input_channels)

    def forward(self, z):
        # Decode
        h, _ = self.rnn(z)
        x_hat = F.sigmoid(self.linear(h))
        return x_hat
