"""Torch-only model definitions shared by training, online use and GPU timing."""
from torch import nn
from torch.nn import functional as F


class Regressor(nn.Module):
    def __init__(self, width, kind='mlp', uncertainty=False):
        super().__init__()
        self.kind = kind
        self.uncertainty = uncertainty
        if kind == 'gru':
            self.encoder = nn.GRU(width, 32, batch_first=True)
            self.net = nn.Linear(32, 6 if uncertainty else 3)
        elif kind == 'tcn':
            self.encoder = nn.ModuleList([nn.Conv1d(width,32,3),nn.Conv1d(32,32,3,dilation=2)])
            self.net = nn.Linear(32,6 if uncertainty else 3)
        else:
            self.net = nn.Sequential(nn.Linear(width,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(),
                                     nn.Linear(64,32), nn.ReLU(), nn.Linear(32,6 if uncertainty else 3))

    def forward(self, x):
        if self.kind == 'gru':
            x = self.encoder(x)[0][:, -1]
        elif self.kind == 'tcn':
            x=x.transpose(1,2)
            x=F.relu(self.encoder[0](F.pad(x,(2,0))))
            x=F.relu(self.encoder[1](F.pad(x,(4,0))))[:,:,-1]
        return self.net(x)
