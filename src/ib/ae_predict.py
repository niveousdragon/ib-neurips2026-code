"""Autoencoder with reconstruction + regression-prediction heads.

Hippocampus probe: T (n_neurons) -> Z (latent_dim) via DRIADA ModularAutoencoder
(the same encoder/decoder as the classification probe in estimator.py),
plus a regression head Z -> y_hat (y_dim) for continuous targets such as the
animal's position.

The reconstruction loss is provided by ModularAutoencoder.compute_loss (a
single "reconstruction" component, weight=1). The prediction loss (MSE on
y_hat vs y) is added by the training loop; see train_ae_predict below.
"""
import gc

import numpy as np
import torch
import torch.nn as nn
from driada.dim_reduction.flexible_ae import ModularAutoencoder


class AEPredict(nn.Module):
    """ModularAutoencoder + regression head on the latent code.

    predict_hidden: 0 -> linear head, >0 -> MLP head with one hidden layer
    of that width.
    """

    def __init__(self, n_neurons: int, latent_dim: int, hidden_dim: int,
                 y_dim: int, predict_hidden: int = 0):
        super().__init__()
        self.ae = ModularAutoencoder(
            input_dim=n_neurons,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            loss_components=[{"name": "reconstruction", "weight": 1.0}],
        )
        if predict_hidden > 0:
            self.predict_head = nn.Sequential(
                nn.Linear(latent_dim, predict_hidden),
                nn.ReLU(),
                nn.Linear(predict_hidden, y_dim),
            )
        else:
            self.predict_head = nn.Linear(latent_dim, y_dim)

    def encode(self, x):
        return self.ae.encode(x)

    def decode(self, z):
        return self.ae.decode(z)

    def forward(self, x):
        z = self.ae.encode(x)
        x_hat = self.ae.decode(z)
        y_hat = self.predict_head(z)
        return z, x_hat, y_hat


def train_ae_predict(T_train, Y_train, T_test, Y_test,
                     latent_dim: int, hidden_dim: int,
                     epochs: int, batch_size: int, lr: float,
                     lambda_predict: float, seed: int = 42,
                     predict_hidden: int = 0):
    """Train AEPredict, return Z on train/test + Y predictions on test.

    Parameters
    ----------
    T_train, T_test : np.ndarray, (n, n_neurons) float32
    Y_train, Y_test : np.ndarray, (n, y_dim) float32

    Returns
    -------
    Z_train      : (n_train, latent_dim) np.ndarray
    Z_test       : (n_test, latent_dim) np.ndarray
    y_hat_test   : (n_test, y_dim) np.ndarray
    recon_mse    : float, test-set reconstruction MSE
    metadata     : dict with "loss_history" mapping "total"/"recon"/"predict"
                   -> list of per-epoch averages
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_neurons = T_train.shape[1]
    y_dim = Y_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AEPredict(n_neurons, latent_dim, hidden_dim, y_dim,
                      predict_hidden=predict_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    T_tr = torch.tensor(T_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(T_tr, Y_tr),
        batch_size=batch_size, shuffle=True,
    )

    history = {"total": [], "recon": [], "predict": []}
    model.train()
    for _ in range(epochs):
        epoch_total = epoch_recon = epoch_predict = 0.0
        n_batches = 0
        for t_b, y_b in loader:
            # Reconstruction loss via ModularAutoencoder's built-in machinery
            recon_loss, _parts = model.ae.compute_loss(t_b)
            # Prediction loss on our own regression head
            z = model.ae.encode(t_b)
            y_hat = model.predict_head(z)
            predict_loss = ((y_hat - y_b) ** 2).mean()
            loss = recon_loss + lambda_predict * predict_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_total += float(loss)
            epoch_recon += float(recon_loss)
            epoch_predict += float(predict_loss)
            n_batches += 1
        history["total"].append(epoch_total / n_batches)
        history["recon"].append(epoch_recon / n_batches)
        history["predict"].append(epoch_predict / n_batches)

    model.eval()
    with torch.no_grad():
        T_te = torch.tensor(T_test, dtype=torch.float32, device=device)
        z_tr = model.ae.encode(T_tr).cpu().numpy()
        z_te = model.ae.encode(T_te).cpu().numpy()
        t_hat_te = model.ae.decode(
            torch.tensor(z_te, dtype=torch.float32, device=device)
        ).cpu().numpy()
        y_hat_te = model.predict_head(
            torch.tensor(z_te, dtype=torch.float32, device=device)
        ).cpu().numpy()
        recon_mse = float(((t_hat_te - T_test) ** 2).mean())

    del model, opt, T_tr, Y_tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return z_tr, z_te, y_hat_te, recon_mse, {"loss_history": history}
