# cloud_run_diff.py
# Author: Christos Gklezos

import torch

import ADWSN.cluster_run as cluster


RUN = False
data_path = "cloud_sample_0.pt"

def loadModel():
    model = torch.load("cloud_diff_ae.pt")
    model.eval()
    return model



def infer(input, model):
    out = model(input)
    loss_fn = torch.nn.MSELoss(reduction="none")
    return loss_fn(input, out)


if RUN:
    # Load Data
    model = loadModel()
    threshhold = 3
    data = torch.load(data_path)
    cluster_model = cluster.loadModel()
    cluster.infer(data, cluster_model)
    data = cluster_model.extract_enc().flatten()
    print(data)
    output = infer(data, model)
    print(output)
    output = torch.sum(output)
    print(output)



