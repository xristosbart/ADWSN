# Filename: data_gen.py
# Author: Christos Gklezos


import torch, torchvision
import math
import pandas as pd
import numpy as np

file = "lab_anomalies_sample.pt"

mode = 4

if mode == 1:

    base = torch.tensor([[x, x+1, x, x, x-0.2, 50-x*0.5] for x in range(5, 45)])
    base = torch.cat((base, torch.tensor([[x, x+1, x, x, x-0.2, 55-x*0.5] for x in range(5, 45)])), 0)
    base = base.unsqueeze(1)
    base = base.expand(80, 2000, 6).flatten(start_dim=0, end_dim=1)

    noise = torch.randn(base.shape)
    sd = torch.tensor([0.1, 0.5, .01, 0.15, 0.2, 5])
    data = noise*sd + base


elif mode == 2:
    b = 20
    base = torch.tensor([[b-10*math.cos(2*math.pi*t/24), b-10*math.cos(2*math.pi*t/24) + 1, b-10*math.cos(2*math.pi*t/24) , b-10*math.cos(2*math.pi*t/24) , b-10*math.cos(2*math.pi*t/24)  - 0.2, 50 - (b-10*math.cos(2*math.pi*t/24)) * 0.5] for t in range(0, 24)])
    base = base.unsqueeze(1).expand(24, 2, 6)
    base[:, 1, 5] = torch.tensor([55 - (b-10*math.cos(2*math.pi*t/24)) * 0.5 for t in range(0, 24)])
    base = base.flatten(start_dim=0, end_dim=1)
    base = base.unsqueeze(1)
    base = base.expand(48, 2000, 6).flatten(start_dim=0, end_dim=1)
    noise = torch.randn(base.shape)
    sd = torch.tensor([0.1, 0.5, .01, 0.15, 0.2, 5])
    data = noise * sd + base
    print(data)

elif mode == 3:
    # Cloud test
    b = 21
    amp = 15
    base = torch.tensor([[b - amp * math.cos(2 * math.pi * t / 24), b - amp * math.cos(2 * math.pi * t / 24) + 1,
                          b - amp * math.cos(2 * math.pi * t / 24), b - amp * math.cos(2 * math.pi * t / 24),
                          b - amp * math.cos(2 * math.pi * t / 24) - 0.2,
                          50 - (b - amp * math.cos(2 * math.pi * t / 24)) * 0.5] for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 15, 15, 15]])
    data = base
    print(data)

elif mode == 4:
    normal = pd.DataFrame(torch.load("lab_sample_data.pt").numpy())
    sample = normal.sample(n=10)
    sample[2] = sample[2] + 4
    anomalies = sample

    sample = normal.sample(n=10)
    sample[3] = 0
    anomalies = anomalies.append(sample)

    sample = normal.sample(n=10)
    sample[4] = sample[4] + 6
    anomalies = anomalies.append(sample)

    sample = normal.sample(n=10)
    sample[4] = sample[4] + 3
    anomalies = anomalies.append(sample)



    data = torch.tensor(anomalies.to_numpy())

torch.save(data, file)
