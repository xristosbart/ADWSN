# Filename: cluster.py
# Author: Christos Gklezos

import torch


# Cluster Model
class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = None
        self.l1 = torch.nn.Linear(n, 4)
        self.l2 = torch.nn.Linear(4, 3)
        self.l3 = torch.nn.Linear(3, 4)
        self.l4 = torch.nn.Linear(4, n)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        y = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(y))
        x = self.l4(x)
        self.enc = y
        return x

    def extract_enc(self):
        return self.enc
