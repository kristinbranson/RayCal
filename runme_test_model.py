import torch
import numpy as np
import torch.nn as nn

class fittingModel(nn.Module):
    def __init__(self):
        super(fittingModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(np.pi/3))
        self.beta = nn.Parameter(torch.tensor(np.pi/6))
        self.gamma = nn.Parameter(torch.tensor(np.pi/10))
        self.center = nn.Parameter(torch.tensor([1.,0.,0.]))
        self.a = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(1.))
        
    def forward(self):
        return self.alpha, self.beta, self.gamma, self.center, self.a, self.b