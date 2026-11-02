"""Classification-autoencoder probe and the relevance estimates built on it.

  1. Train a ModularAutoencoder with reconstruction + classification loss.
  2. Encode held-out data into the latent space Z.
  3. Estimate I(Z; Y), a lower bound on I(T; Y): from the temperature-scaled
     cross-entropy of the classification head (``_mi_from_ce``, used for
     cortex) or with KSG / Gaussian-copula estimators on Z.
"""
import gc
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.utils.data
from driada.dim_reduction.flexible_ae import ModularAutoencoder
from driada.information.gcmi import mi_model_gd
from driada.information.ksg import nonparam_mi_cd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

LN2 = np.log(2)


@dataclass
class AEConfig:
    """Hyperparameters for the classification autoencoder."""
    latent_dim: int = 16
    hidden_dim: int = 256
    ae_epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 128
    class_weight: float = 1.0
    class_hidden: int = 0  # 0 = linear head, >0 = MLP with this hidden dim
    seed: int = 42


@dataclass
class IBResult:
    """Results of IB estimation for one (layer, epoch) pair."""
    I_zy: float = np.nan       # I(Z; Y) via KSG/GCMI, bits
    I_ce: float = np.nan       # I(Z; Y) via cross-entropy: H(Y) - CE, bits
    I_ind: float = np.nan      # sum of per-neuron I(t_i; Y), bits
    H_gauss: float = np.nan    # Gaussian upper bound on H(T), bits
    acc_z: float = np.nan      # accuracy from latent code Z
    acc_full: float = np.nan   # accuracy from full activity T (LogReg)
    recon_mse: float = np.nan  # reconstruction MSE
    n_neurons: int = 0
    I_zy_shuffles: np.ndarray = field(default_factory=lambda: np.array([]))  # null distribution


def train_ae_classifier(act_train, labels_train, act_test, labels_test,
                        n_classes, cfg=None):
    """Train classification AE on train, return latent codes, accuracy, and logits on test.

    Returns:
        Z_test: (n_test, latent_dim) latent codes
        acc_z: classification accuracy from Z
        recon_mse: reconstruction MSE on test
        logits: (n_test, n_classes) raw classifier outputs
    """
    if cfg is None:
        cfg = AEConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    N = act_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_components = [
        {"name": "reconstruction", "weight": 1.0},
        {"name": "classification", "weight": cfg.class_weight,
         "num_classes": n_classes},
    ]
    if cfg.class_hidden > 0:
        loss_components[1]["hidden_dim"] = cfg.class_hidden

    model = ModularAutoencoder(
        input_dim=N,
        latent_dim=cfg.latent_dim,
        hidden_dim=min(cfg.hidden_dim, N * 2),
        loss_components=loss_components,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_tensor = torch.tensor(act_train, dtype=torch.float32, device=device)
    train_labels = torch.tensor(labels_train, dtype=torch.long, device=device)

    dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.ae_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            total_loss, _ = model.compute_loss(batch_x, labels=batch_y)
            total_loss.backward()
            optimizer.step()

    model.eval()
    test_tensor = torch.tensor(act_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        Z_train = model.encode(train_tensor).cpu().numpy()
        Z_test = model.encode(test_tensor).cpu().numpy()
        recon = model(test_tensor).cpu().numpy()
        recon_mse = float(np.mean((act_test - recon) ** 2))

        cls_loss = model.losses[1]
        logits_t = cls_loss.classifier(model.encode(test_tensor))
        logits = logits_t.cpu().numpy()
        preds = logits_t.argmax(dim=1).cpu().numpy()
        acc_z = float(np.mean(preds == labels_test))

    del model, optimizer, train_tensor, train_labels, test_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Z_train, Z_test, acc_z, recon_mse, logits


def estimate_ib(act_train, labels_train, act_test, labels_test,
                n_classes, cfg=None, mi_method="ksg", n_shuffles=0):
    """Full IB estimation for one population snapshot.

    Args:
        act_train: (n_train, N) neural activity, train split
        labels_train: (n_train,) discrete labels
        act_test: (n_test, N) neural activity, test split
        labels_test: (n_test,) discrete labels
        n_classes: number of classes
        cfg: AEConfig
        mi_method: "ksg" (nonparametric, needs large N) or "gcmi" (Gaussian copula, small-sample robust)
        n_shuffles: number of label-shuffle iterations for null distribution (0 = skip)

    Returns:
        IBResult with all metrics
    """
    if cfg is None:
        cfg = AEConfig()
    result = IBResult(n_neurons=act_test.shape[1])

    # Per-neuron MI (test set)
    # For discrete activations (spike counts): exact MI via contingency table
    # For continuous activations (dF/F): Gaussian copula MI
    if np.issubdtype(act_test.dtype, np.integer):
        mi_per_neuron = np.array([
            mutual_info_score(labels_test, act_test[:, k]) / LN2
            for k in range(act_test.shape[1])
        ])
    else:
        mi_per_neuron = np.array([
            mi_model_gd(act_test[:, k], labels_test)
            for k in range(act_test.shape[1])
        ])
    result.I_ind = float(mi_per_neuron.sum())

    # Gaussian entropy upper bound
    result.H_gauss = gaussian_entropy(act_test)

    # Baseline accuracy (LogReg, train -> test)
    result.acc_full = baseline_accuracy(
        act_train, labels_train, act_test, labels_test)

    # AE + MI
    _Z_train, Z_test, acc_z, recon_mse, logits = train_ae_classifier(
        act_train, labels_train, act_test, labels_test,
        n_classes=n_classes, cfg=cfg)

    result.acc_z = acc_z
    result.recon_mse = recon_mse
    result.I_zy = _estimate_mi(Z_test, labels_test, mi_method)

    # I(Z;Y) via cross-entropy: H(Y) - CE
    result.I_ce = _mi_from_ce(logits, labels_test, n_classes)

    # Shuffle control: permute labels, recompute MI on same Z
    if n_shuffles > 0:
        rng = np.random.RandomState(cfg.seed)
        shuf_vals = np.empty(n_shuffles)
        for i in range(n_shuffles):
            y_shuf = rng.permutation(labels_test)
            shuf_vals[i] = _estimate_mi(Z_test, y_shuf, mi_method)
        result.I_zy_shuffles = shuf_vals

    return result


def _find_temperature(logits, labels):
    """Find optimal temperature T that minimizes NLL on the given data.

    Uses simple grid search + refinement (1 parameter, fast).
    """
    from scipy.optimize import minimize_scalar

    def nll(log_T):
        T = np.exp(log_T)
        scaled = logits / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(labels)), labels].mean()

    res = minimize_scalar(nll, bounds=(-3, 5), method='bounded')
    return float(np.exp(res.x))


def _mi_from_ce(logits, labels, n_classes):
    """I(Z;Y) = H(Y) - CE(Y, softmax(logits/T)), in bits.

    Temperature T is found by minimizing NLL (Guo et al. 2017).
    """
    T = _find_temperature(logits, labels)
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    ce_nats = -log_probs[np.arange(len(labels)), labels].mean()
    ce_bits = ce_nats / LN2
    H_Y = np.log2(n_classes)
    return float(max(0.0, H_Y - ce_bits))


def _estimate_mi(Z, labels, mi_method):
    """Estimate I(Z; Y) with error handling."""
    try:
        if mi_method == "gcmi":
            return float(mi_model_gd(Z.T, labels))
        else:
            return float(nonparam_mi_cd(Z, labels, k=5, base=2.0))
    except np.linalg.LinAlgError:
        return 0.0


def gaussian_entropy(act):
    """H_gauss(T) = 1/2 sum log2(2*pi*e * lambda_k), bits."""
    cov = np.cov(act, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return float(0.5 * np.sum(np.log2(2 * np.pi * np.e * eigenvalues)))


def baseline_accuracy(act_train, labels_train, act_test, labels_test):
    """Logistic regression accuracy: train -> test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, solver='lbfgs', n_jobs=-1)
        lr.fit(act_train, labels_train)
        return float(lr.score(act_test, labels_test))


def stratified_subsample(act, labels, n_total, n_classes, seed=42):
    """Stratified subsample: n_total/n_classes per class."""
    n_per_class = n_total // n_classes
    idx = []
    rng = np.random.RandomState(seed)
    for c in range(n_classes):
        mask = np.where(labels == c)[0]
        chosen = rng.choice(mask, size=min(n_per_class, len(mask)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return act[idx], labels[idx]
