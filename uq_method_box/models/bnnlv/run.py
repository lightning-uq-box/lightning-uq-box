import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import energy_function

target_f = lambda x: (x**2 + np.sin(x)) / 5.0
from bnn import BNN

N = 250
X = torch.randn(N, 1)
Y = target_f(X) + torch.randn(N, 1) * 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {"device": device, "graph": [30, 30]}
model = BNN(1, 1, config=config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X = X.to(device)
Y = Y.to(device)
## train model on random mini-batches
batches_per_epoch = 2_048
epochs = 10
for i in range(epochs):
    loss_val = 0

    for j in range(batches_per_epoch):
        optimizer.zero_grad()
        idx = np.random.choice(N, 32)
        x = X[idx]
        y = Y[idx]
        y_pred = model(x)
        loss_terms = model.get_loss_terms()
        loss = energy_function(y, y_pred, loss_terms, N, alpha=1.0)
        loss.backward()
        loss_val += loss.item()

        optimizer.step()
    print(f"Epoch {i}: {loss_val/batches_per_epoch}")

# test on a grid and plot the preidctions and the uncertainty
x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
y_test = target_f(x_test)
x_test = torch.tensor(x_test).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

y_pred = model(x_test).mean

to_plot = lambda x: x.cpu().detach().numpy().ravel()
import matplotlib.pyplot as plt

plt.plot(to_plot(x_test), to_plot(y_test), label="target")
plt.plot(to_plot(x_test), to_plot(y_pred.mean(1)), label="mean")
plt.fill_between(
    to_plot(x_test),
    to_plot(y_pred.mean(1) - y_pred.std(1)),
    to_plot(y_pred.mean(1) + y_pred.std(1)),
    alpha=0.5,
    label="stddev",
)
plt.legend()
plt.show()
