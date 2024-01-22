import torch
import torch.nn as nn

class FFN(torch.nn.Module):
    def __init__(self, input_dim=2):
        super(FFN, self).__init__()
        ### TODO: define and initialize some layers with weights
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, coord):
        out = self.layers(coord)
        return out
    
class Sin(torch.nn.Module):
  def forward(self, x):
    return torch.sin(x)

class FFN_Sin(torch.nn.Module):
    def __init__(self, input_dim=2):
        super(FFN_Sin, self).__init__()
        ### TODO: define and initialize some layers with weights
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            Sin(),
            nn.Linear(1024, 512),
            Sin(),
            nn.Linear(512, 256),
            Sin(),
            nn.Linear(256, 3),
        )

    def forward(self, coord):
        out = self.layers(coord)
        return out