# Filename: cluster_train.py
# Author: Christos Gklezos

import torch, torchvision
import numpy as np
import csv

import ADWSN.cluster as cluster

# Train Data
file = 'sample_3.pt'#"lab_sample_data_1.pt"

# Parameters
learning_rate = 1e-3
epochs = 50
data = torch.load(file)
print(data.shape)
length_t = int(len(data)-40)

train_set, val_set = torch.utils.data.random_split(data.float(), [length_t, len(data) - length_t])
trainset = torch.utils.data.DataLoader(train_set, batch_size=40, shuffle=True)
valset = torch.utils.data.DataLoader(val_set, batch_size=40, shuffle=True)


# Model Loading
model = cluster.Net(6)

# Optimizer Setup
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction="sum")

# Train Loop
def train(epochs, optimizer, model, loss_fn, trainset, valset):

    for epoch in range(epochs):
        for batch in trainset:
            out = model(batch)
            loss = loss_fn(out, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f"Epoch {epoch},\tTraining Loss: {loss.item():.4f}")
        for batch in valset:
            out = model(batch)
            vloss = loss_fn(out, batch)
        #print(f"Epoch {epoch},\tValidation Loss: {loss.item():.4f}")
        print(f" {epoch},\t{loss.item():.4f},\t{vloss.item()}")



train(epochs,
      optimizer,
      model,
      loss_fn,
      trainset,
      valset)

# Saving Model
torch.save(model, "cluster_ae.pt")