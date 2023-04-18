import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
from loss import energy_function
from bnn import BNNLV


def f(x):
    return 7 * np.sin(x) + 3 * np.abs(np.cos(x / 2)) * np.random.randn()


N = 750
X = np.zeros((N, 1))
Y = np.zeros((N, 1))

for k in range(N):
    rnd = np.random.rand()
    if rnd < 1 / 3.0:
        X[k] = np.random.normal(loc=-4, scale=2.0 / 5.0)
    else:
        if rnd < 2.0 / 3.0:
            X[k] = np.random.normal(loc=0.0, scale=0.9)
        else:
            X[k] = np.random.normal(loc=4.0, scale=2.0 / 5.0)
    Y[k] = f(X[k])

mean_X = X.mean()
std_X = X.std()
mean_Y = Y.mean()
std_Y = Y.std()
X_n = torch.tensor((X - mean_X) / std_X).float()
Y_n = torch.tensor((Y - mean_Y) / std_Y).float()

X_ind = torch.tensor(np.arange(N)).float().reshape(-1, 1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {"device": device, "graph": [50, 50]}
model = BNNLV(1, 1, N, config=config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_n = X_n.to(device)
Y_n = Y_n.to(device)
X_ind = X_ind.to(device)
## train model on random mini-batches
batches_per_epoch = 4_096
epochs = 10
loss_val = 0
print(
    f"Epoch {-1}: {0} lv_mu: {model.lv.z_mu.cpu().detach().mean()} lv_std: {F.softplus(model.lv.z_log_sigma.mean()).cpu().detach()}  log noise: {model.log_aleatoric_std.cpu().detach()}"
)

for i in range(epochs):
    loss_val = 0

    for j in range(batches_per_epoch):
        optimizer.zero_grad()
        idx = np.random.choice(N, 64)
        x = X_n[idx]
        ind = X_ind[idx]
        y = Y_n[idx]

        y_pred = model([x, ind])
        loss_terms = model.get_loss_terms()
        loss = energy_function(y, y_pred, loss_terms, N, alpha=1.0)
        loss.backward()
        loss_val += loss.item()

        optimizer.step()
    print(
        f"Epoch {i}: {loss_val/batches_per_epoch} lv_mu: {model.lv.z_mu.mean().cpu().detach()} lv_std: {F.softplus(model.lv.z_log_sigma).mean().cpu().detach()} log noise: {model.log_aleatoric_std.cpu().detach()}"
    )

# test on a grid and plot the preidctions and the uncertainty
x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
y_test = np.array([f(x) for x in x_test]).reshape(-1, 1)
x_test = torch.tensor((x_test - mean_X) / std_X).float().to(device)
y_test = torch.tensor((y_test - mean_Y) / std_Y).float().to(device)

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

random_w = 500
random_z = 500
N_pool = 100
X_pool = np.linspace(-4, 4, N_pool)[:, None]
n_samples = 25
n_runs = int(random_w / n_samples)
upscale = 100
z = np.random.randn(random_z * upscale)

X_pool_in = (np.tile(X_pool[None, :, :], [25, 1, 1]) - mean_X) / std_X

out_hy = np.zeros((N_pool, random_w * upscale))
for k in range(int((upscale * random_w) / n_samples)):
    for layer in model.net:
        layer.fix_randomness()
    inp_n = np.tile(
        z[k * n_samples : (k + 1) * n_samples][:, None, None], [1, N_pool, 1]
    )

    pred = model(
        [
            torch.tensor(X_pool_in).float().to(device),
            torch.tensor(inp_n).float().to(device),
        ],
        training=False,
    ).sample()
    out_hy[:, k * n_samples : (k + 1) * n_samples] = (
        pred.cpu().detach().numpy()[:, :, 0]
    )


out = np.zeros((N_pool, random_w, random_z))
for k in range(int(random_w / n_samples)):
    print(k)
    for layer in model.net:
        layer.fix_randomness()
    for j in range(random_z):
        inp_n = np.tile(z[j : j + 1][None, :, None], [n_samples, N_pool, 1])

        pred = model(
            [
                torch.tensor(X_pool_in).float().to(device),
                torch.tensor(inp_n).float().to(device),
            ],
            training=False,
        ).sample()
        out[:, k * n_samples : (k + 1) * n_samples, j] = (
            pred.cpu().detach().numpy()[:, :, 0]
        )


def entropy(x):
    return 0.5 * np.log(2 * np.pi * np.var(x)) + 0.5


HY = np.array([entropy(out_hy[k]) for k in range(100)])
HYGW = np.array(
    [np.mean([entropy(out[k, j, :]) for j in range(random_w)]) for k in range(100)]
)
plt.figure()
plt.plot(X_pool[:, 0], HY)
plt.plot(X_pool[:, 0], HYGW)
plt.plot(X_pool[:, 0], HY - HYGW)
plt.show()
