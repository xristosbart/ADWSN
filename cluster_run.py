# Filename: cluster_run.py
# Author: Christos Gklezos

import torch

import csv

RUN = False
data_path = "sample_14.pt"
date = "05_01_21"

# sample_path = "sample_0.csv"
#
# sample_np = np.loadtxt(sample_path, dtype=np.float32, delimiter=",", skiprows=2)
# reader = csv.reader(open(sample_path), delimiter=',')

# col_list = next(reader)
# anomaly = next(reader)


def loadModel():
    model = torch.load("cluster_ae.pt")
    model.eval()
    return model


model = loadModel()
loss_fn = torch.nn.MSELoss(reduction="none")


def infer(input, model):
    out = model(input)
    return loss_fn(input, out)


def extract_to_file(data, date, timestamp):
    torch.save(data, f"{date}/{timestamp}.pt")


if RUN:
    # Load Data
    model = loadModel()
    threshhold = 3
    data = torch.load(data_path)
    data = torch.split(data, 4000, 0)

    for timestamp, batch in enumerate(data):
        output = torch.sum(infer(batch, model), 1)
        intermediates = model.extract_enc()
        extract_to_file(intermediates, date, timestamp)
        print(torch.sum(torch.gt(output, threshhold)).item())
