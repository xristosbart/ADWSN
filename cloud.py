# Filename: cloud.py
# Author: Christos Gklezos

import torch


# Cloud Model
class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = None
        self.l1 = torch.nn.Linear(n, 60)
        self.l2 = torch.nn.Linear(60, 40)
        self.l3 = torch.nn.Linear(40, 30)
        self.l4 = torch.nn.Linear(30, 20)
        self.l5 = torch.nn.Linear(20, 3)
        self.l6 = torch.nn.Linear(3, 20)
        self.l7 = torch.nn.Linear(20, 30)
        self.l8 = torch.nn.Linear(30, 40)
        self.l9 = torch.nn.Linear(40, 60)
        self.l10 = torch.nn.Linear(60, n)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(self.l4(x))
        x = torch.tanh(self.l5(x))
        x = torch.tanh(self.l6(x))
        x = torch.tanh(self.l7(x))
        x = torch.tanh(self.l8(x))
        x = torch.tanh(self.l9(x))
        x = self.l10(x)
        return x
