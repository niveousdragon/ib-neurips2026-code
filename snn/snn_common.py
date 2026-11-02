"""Constants, data layout and loaders shared by the SNN scripts.

Raw data (not redistributed) lives under IB_DATA_SNN:

    Activity/CIFAR train labels.npz, Activity/CIFAR test labels.npz
    Activity/SJ-SNN-50/iter <E>/test/SJ-SNN act iter <E> test layer <L>.npy.npz
    Activity/SJ-SNN-50/iter <E>/train/SJ-SNN act iter <E> train layer <L>.npz
    SJ-SNN-T50/SJ-SNN evolution data full 3.yaml     (MEI results)
    SJ-SNN-T50/SJ-SNN T50 test acc.npz               (test-accuracy curve)
    cifar/cifar-10-batches-py/test_batch             (CIFAR-10 images, 09_fetch_cifar.py)

Each activity file holds one array of shape (T=50 timesteps, n_images, N channels);
the analysis uses the spike count per channel, i.e. the sum over timesteps.
"""
import gc

import numpy as np
import torch

from ib.paths import data_dir

LAYERS = [
    "sn1",
    "layer1.0.sn1", "layer1.0.sn2", "layer1.1.sn1", "layer1.1.sn2",
    "layer2.0.sn1", "layer2.0.sn2", "layer2.1.sn1", "layer2.1.sn2",
    "layer3.0.sn1", "layer3.0.sn2", "layer3.1.sn1", "layer3.1.sn2",
    "layer4.0.sn1", "layer4.0.sn2", "layer4.1.sn1", "layer4.1.sn2",
]
EPOCHS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175,
          200, 225, 250, 275, 300, 400, 500, 600, 700, 800, 900]
# First spike layer of the five blocks shown in the main information-plane figure.
PANEL_LAYERS = ["layer2.1.sn1", "layer3.0.sn1", "layer3.1.sn1",
                "layer4.0.sn1", "layer4.1.sn1"]

N_CLASSES = 10
LN2 = np.log(2)

# Classification-autoencoder probe (paper, Supplementary "Autoencoder hyperparameters").
TRAIN_SUBSAMPLE = 10000  # class-stratified subsample of the 50 000 training images
LATENT_DIM = 16
HIDDEN_DIM = 256
AE_EPOCHS = 50
AE_LR = 1e-3
AE_BATCH = 128
CLASS_WEIGHT = 1.0
SUBSAMPLE_SEED = 42

# Relative locations under IB_DATA_SNN.
ACTIVITY_SUBDIR = "Activity"
SNN_SUBDIR = "SJ-SNN-50"
MEI_YAML = "SJ-SNN-T50/SJ-SNN evolution data full 3.yaml"
TEST_ACC_NPZ = "SJ-SNN-T50/SJ-SNN T50 test acc.npz"
CIFAR_SUBDIR = "cifar"


def activity_root():
    return data_dir("snn") / ACTIVITY_SUBDIR


def snn_dir():
    return activity_root() / SNN_SUBDIR


def require_activity():
    """Stop with a clear message if the raw activity is not available."""
    if not snn_dir().exists():
        raise SystemExit(
            f"SNN activity not found at {snn_dir()}. Set IB_DATA_SNN to the folder "
            f"that contains {ACTIVITY_SUBDIR}/{SNN_SUBDIR} (see snn_common.py for the layout)."
        )


def short_name(layer):
    """'layer3.1.sn1' -> 'L3.1.sn1'."""
    return "sn1" if layer == "sn1" else layer.replace("layer", "L")


def load_labels(split="test"):
    return np.load(activity_root() / f"CIFAR {split} labels.npz")["arr_0"]


def load_activity(epoch, layer, split="test"):
    """Spike counts (n_images, N) of one sublayer at one checkpoint, or None if absent."""
    suffix = ".npz" if split == "train" else ".npy.npz"
    path = snn_dir() / f"iter {epoch}" / split / f"SJ-SNN act iter {epoch} {split} layer {layer}{suffix}"
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    key = "arr_0" if "arr_0" in data else "a"
    return data[key].sum(axis=0)


def stratified_subsample(act, labels, n_total=TRAIN_SUBSAMPLE):
    """n_total / 10 images drawn from each class, fixed seed."""
    n_per_class = n_total // N_CLASSES
    idx = []
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    for c in range(N_CLASSES):
        mask = np.where(labels == c)[0]
        idx.append(rng.choice(mask, size=min(n_per_class, len(mask)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return act[idx], labels[idx]


def gaussian_entropy(act):
    """H_gauss(T) = 0.5 * sum_k log2(2 pi e lambda_k), bits.

    lambda_k are eigenvalues of the sample covariance of T; eigenvalues at the
    level of numerical noise (<= 1e-10) are excluded to avoid log(0).
    """
    eigenvalues = np.linalg.eigvalsh(np.cov(act, rowvar=False))
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return 0.5 * np.sum(np.log2(2 * np.pi * np.e * eigenvalues))


def train_ae_classifier(act_train, labels_train, act_test, labels_test,
                        latent_dim=LATENT_DIM, class_weight=CLASS_WEIGHT,
                        ae_epochs=AE_EPOCHS):
    """Train the reconstruction + classification autoencoder on train, encode test.

    Returns (Z_test, accuracy of the classification head on test, reconstruction MSE
    on test). With class_weight=0 the head is present but untrained, so its accuracy
    is meaningless; only Z_test is used in that case.
    """
    from driada.dim_reduction.flexible_ae import ModularAutoencoder

    n_in = act_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModularAutoencoder(
        input_dim=n_in,
        latent_dim=latent_dim,
        hidden_dim=min(HIDDEN_DIM, n_in * 2),
        loss_components=[
            {"name": "reconstruction", "weight": 1.0},
            {"name": "classification", "weight": class_weight, "num_classes": N_CLASSES},
        ],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    train_tensor = torch.tensor(act_train, dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels_train, dtype=torch.long).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor, train_labels),
        batch_size=AE_BATCH, shuffle=True)

    model.train()
    for _ in range(ae_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            total_loss, _ = model.compute_loss(batch_x, labels=batch_y)
            total_loss.backward()
            optimizer.step()

    model.eval()
    test_tensor = torch.tensor(act_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = model.encode(test_tensor)
        z_test = z.cpu().numpy()
        recon_mse = float(np.mean((act_test - model(test_tensor).cpu().numpy()) ** 2))
        preds = model.losses[1].classifier(z).argmax(dim=1).cpu().numpy()
        acc_z = float(np.mean(preds == labels_test))

    del model, train_tensor, train_labels, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return z_test, acc_z, recon_mse


def fit_vertex(y, half_window=4):
    """Vertex of a parabola fitted to a layer profile around its argmax.

    The fit uses the layers within +/- half_window of argmax(y): a global parabola
    would be skewed by the long, gentle decay of the profile through L3-L4.
    Returns (vertex_x, concavity_a); vertex_x is NaN when the fit has no interior
    maximum.
    """
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan
    am = int(np.argmax(np.where(m, y, -np.inf)))
    lo = max(0, am - half_window)
    hi = min(len(x), am + half_window + 1)
    sub_x, sub_y = x[lo:hi], y[lo:hi]
    sm = np.isfinite(sub_y)
    if sm.sum() < 4:
        return np.nan, np.nan
    a, b, _ = np.polyfit(sub_x[sm], sub_y[sm], deg=2)
    if abs(a) < 1e-10:
        return np.nan, np.nan
    xstar = -b / (2 * a)
    if a >= 0:
        return np.nan, a
    if xstar < sub_x[sm].min() or xstar > sub_x[sm].max():
        return np.nan, a
    return xstar, a


AUX_NAMES = ["brightness", "contrast", "mean_R", "mean_G", "mean_B", "hf_energy"]


def cifar_test_images():
    """CIFAR-10 test images as floats in [0, 1], shape (10000, 3, 32, 32)."""
    import pickle

    path = data_dir("snn") / CIFAR_SUBDIR / "cifar-10-batches-py" / "test_batch"
    if not path.exists():
        raise SystemExit(f"{path} not found; run 09_fetch_cifar.py first.")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).astype(np.float64) / 255.0


def image_statistics(img):
    """Six z-scored statistics per image, columns in the order of AUX_NAMES.

    Brightness and contrast are the mean and SD of the grey image, hf_energy the mean
    squared 4-neighbour Laplacian of the grey image. None of them is the class label
    and none depends on the network.
    """
    n = len(img)
    gray = img.mean(1)
    lap = (gray[:, 1:-1, 2:] + gray[:, 1:-1, :-2] + gray[:, 2:, 1:-1]
           + gray[:, :-2, 1:-1] - 4 * gray[:, 1:-1, 1:-1])
    aux = np.column_stack([
        gray.reshape(n, -1).mean(1), gray.reshape(n, -1).std(1),
        img[:, 0].reshape(n, -1).mean(1), img[:, 1].reshape(n, -1).mean(1),
        img[:, 2].reshape(n, -1).mean(1), (lap ** 2).reshape(n, -1).mean(1)])
    return (aux - aux.mean(0)) / aux.std(0)
