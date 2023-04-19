import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from bnn import BNNLV
from loss import energy_function
from numpy import pi
from scipy import ndimage
from scipy.linalg import det
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors

problem = "bimodal"


if problem == "heteroscedastic":

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

if problem == "bimodal":
    N = 750
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))

    def f(x):
        z = np.random.randint(0, 2)
        return z * 10 * np.cos(x) + (1 - z) * 10 * np.sin(x) + np.random.randn()

    for k in range(N):
        x = np.random.exponential(0.5) - 0.5
        while x < -0.5 or x > 2:
            x = np.random.exponential(0.5) - 0.5
        X[k] = x
        Y[k] = f(X[k])


plt.figure()
plt.plot(X, Y, ".")

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_n = X_n.to(device)
Y_n = Y_n.to(device)
X_ind = X_ind.to(device)
## train model on random mini-batches
batches_per_epoch = 4_096
epochs = 25
loss_val = 0
print(f"Epoch {-1}: {0} lv_mu: {model.lv.z_mu.cpu().detach().mean()}", end=" ")
print(f"lv_std: {F.softplus(model.lv.z_log_sigma.mean()).cpu().detach()}", end=" ")
print(f"log noise: {model.log_aleatoric_std.cpu().detach().numpy()}")


for i in range(epochs):
    loss_val = 0

    for j in range(batches_per_epoch):
        idx = np.random.choice(N, 64)
        x = X_n[idx]
        ind = X_ind[idx]
        y = Y_n[idx]

        y_pred = model([x, ind])
        loss_terms = model.get_loss_terms()
        loss = energy_function(y, y_pred, loss_terms, N, alpha=1.0)
        optimizer.zero_grad()
        loss.backward()
        loss_val += loss.item()

        optimizer.step()
    print(f"Epoch {i} Loss: {loss_val/batches_per_epoch} lv_mu: {model.lv.z_mu.cpu().detach().mean()}", end=" ")
    print(f"lv_std: {F.softplus(model.lv.z_log_sigma.mean()).cpu().detach()}", end=" ")
    print(f"log noise: {model.log_aleatoric_std.cpu().detach().numpy()}")


# test on a grid and plot the preidctions and the uncertainty
x_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_test = np.array([f(x) for x in x_test]).reshape(-1, 1)
x_test = torch.tensor((x_test - mean_X) / std_X).float().to(device)
y_test = torch.tensor((y_test - mean_Y) / std_Y).float().to(device)

y_pred = torch.cat([model(x_test).mean for _ in range(10)], dim=1)


to_plot = lambda x: x.cpu().detach().numpy().ravel()

plt.figure()
plt.plot(to_plot(x_test), to_plot(y_test), ".", label="target")
plt.plot(to_plot(x_test), to_plot(y_pred.mean(1)), label="mean")
for k in range(y_pred.shape[1]):
    plt.plot(to_plot(x_test), to_plot(y_pred[:, k]), "k.")
plt.fill_between(
    to_plot(x_test),
    to_plot(y_pred.mean(1) - y_pred.std(1)),
    to_plot(y_pred.mean(1) + y_pred.std(1)),
    alpha=0.5,
    label="stddev",
)
plt.legend()

random_w = 500
random_z = 500
N_pool = 100
X_pool = np.linspace(X.min(), X.max(), N_pool)[:, None]

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


def nearest_distances(X, k=1):
    """
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    """
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def entropy(X, k=20):
    """Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    """

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k)  # squared distances
    n, d = X.shape
    volume_unit_ball = (pi ** (0.5 * d)) / gamma(0.5 * d + 1)
    """
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    """
    return (
        d * np.mean(np.log(r + np.finfo(X.dtype).eps))
        + np.log(volume_unit_ball)
        + psi(n)
        - psi(k)
    )


5


HY = np.array([entropy(out_hy[k][:, None]) for k in range(100)])
HYGW = np.array(
    [
        np.mean([entropy(out[k, j, :][:, None]) for j in range(random_w)])
        for k in range(100)
    ]
)
plt.figure()
plt.plot(X_pool[:, 0], HY)
plt.plot(X_pool[:, 0], HYGW)
plt.plot(X_pool[:, 0], HY - HYGW)
plt.show()
