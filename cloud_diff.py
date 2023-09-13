# Filename: cloud_diff.py
# Author: Christos Gklezos

import torch


# Cloud Model
class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = None
        self.l1 = torch.nn.Linear(n, 3)
        self.l2 = torch.nn.Linear(3, 2)
        self.l3 = torch.nn.Linear(2, 3)
        self.l4 = torch.nn.Linear(3, n)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = self.l4(x)
        return x
