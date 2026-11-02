"""AEPredict: forward-pass shapes + overfit small batch."""
import numpy as np
import torch

from ib.ae_predict import AEPredict


def test_forward_pass_shapes():
    torch.manual_seed(0)
    model = AEPredict(n_neurons=50, latent_dim=8, hidden_dim=32, y_dim=2)
    x = torch.randn(16, 50)
    Z, x_hat, y_hat = model(x)
    assert Z.shape == (16, 8)
    assert x_hat.shape == (16, 50)
    assert y_hat.shape == (16, 2)


def test_ae_can_overfit_small_batch():
    """AE with two heads should drive joint loss down on a fixed small batch."""
    torch.manual_seed(0)
    N, d, H, yd = 20, 4, 16, 2
    model = AEPredict(n_neurons=N, latent_dim=d, hidden_dim=H, y_dim=yd)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, N)
    y = torch.randn(8, yd)

    losses = []
    for _ in range(300):
        Z, x_hat, y_hat = model(x)
        loss = ((x_hat - x) ** 2).mean() + ((y_hat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Overfit test: loss drops by > 10x
    assert losses[-1] < losses[0] / 10


from ib.ae_predict import train_ae_predict


def test_train_ae_predict_drives_loss_down_on_synthetic():
    """Linear Y = T @ W. Training should learn to predict Y from Z."""
    rng = np.random.default_rng(0)
    N, n_train, y_dim = 20, 500, 2
    T_train = rng.standard_normal((n_train, N)).astype(np.float32)
    W = rng.standard_normal((N, y_dim)).astype(np.float32)
    Y_train = (T_train @ W) + 0.05 * rng.standard_normal(
        (n_train, y_dim)).astype(np.float32)
    T_test = rng.standard_normal((200, N)).astype(np.float32)
    Y_test = (T_test @ W).astype(np.float32)

    Z_train, Z_test, y_hat_test, recon_mse, metadata = train_ae_predict(
        T_train, Y_train, T_test, Y_test,
        latent_dim=8, hidden_dim=32, epochs=200,
        batch_size=64, lr=1e-3, lambda_predict=1.0, seed=0,
    )
    # Shapes
    assert Z_train.shape == (n_train, 8)
    assert Z_test.shape == (200, 8)
    assert y_hat_test.shape == (200, y_dim)
    # Metadata has loss history
    assert "loss_history" in metadata
    assert "total" in metadata["loss_history"]
    # Prediction quality: test MSE(y_hat, Y_test) should be lower than var(Y_test)
    pred_mse = float(((y_hat_test - Y_test) ** 2).mean())
    var_y = float(Y_test.var())
    assert pred_mse < var_y / 2, f"pred_mse={pred_mse:.3f}, var(Y)={var_y:.3f}"
