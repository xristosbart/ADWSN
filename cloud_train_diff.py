# Filename: cloud_train_diff.py
# Author: Christos Gklezos


import torch

# Load Model

import ADWSN.cloud_diff as cloud



# Load Data
data = torch.empty(4000, 0, 3)
d = 1
while True:
    date = f"0{d}_01_21"

    try:
        for i in range(0, 24):
            t = torch.load(f"{date}/{i}.pt")
            t = torch.unsqueeze(t, 1)
            data = torch.cat((data, t), 1)
    except:
        break
    d += 1

# print(data[1])
# torch.save(data, "cloud_train_data.pt")
# exit(0)

data = torch.reshape(data, (4000, 5, 24, 3))
data = data[:, :, 1:, :] - data[:, :, :-1, :]
data = torch.reshape(data, (4000, 115, 3))
data = torch.reshape(data, (4000*115, 3))

# print(data.shape)
# print(data[1])

# Training Cycle
learning_rate = 1e-3
epochs = 10

trainset = torch.utils.data.DataLoader(data[:115000], batch_size=200, shuffle=True)

# Model
model = cloud.Net(3)

# Optimizer Setup
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction="sum")

# Train Loop
def train(epochs, optimizer, model, loss_fn, trainset):
    for epoch in range(epochs):
        for batch in trainset:
            out = model(batch)
            loss = loss_fn(out, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch},\tTraining Loss: {loss.item():.4f}")


train(epochs,
      optimizer,
      model,
      loss_fn,
      trainset)

# Saving Model
torch.save(model, "cloud_diff_ae.pt")