import numpy as np
import gzip
import os
import pickle
import urllib.request

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".numpyml_data")


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _download(url, filename):
    _ensure_dir()
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
    return path


def _load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0


def _load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_fashion_mnist():
    base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {k: _download(base + v, "fmnist_" + v) for k, v in files.items()}
    X_train = _load_mnist_images(paths["train_images"])
    y_train = _load_mnist_labels(paths["train_labels"])
    X_test = _load_mnist_images(paths["test_images"])
    y_test = _load_mnist_labels(paths["test_labels"])
    return (X_train, y_train), (X_test, y_test)


def load_mnist():
    base = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {k: _download(base + v, "mnist_" + v) for k, v in files.items()}
    X_train = _load_mnist_images(paths["train_images"])
    y_train = _load_mnist_labels(paths["train_labels"])
    X_test = _load_mnist_images(paths["test_images"])
    y_test = _load_mnist_labels(paths["test_labels"])
    return (X_train, y_train), (X_test, y_test)


def load_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = _download(url, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(CACHE_DIR, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(CACHE_DIR)
    X_train, y_train = [], []
    for i in range(1, 6):
        with open(os.path.join(extract_dir, f"data_batch_{i}"), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        X_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])
    X_train = np.concatenate(X_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    y_train = np.array(y_train)
    with open(os.path.join(extract_dir, "test_batch"), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X_test = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    y_test = np.array(batch[b'labels'])
    return (X_train, y_train), (X_test, y_test)


def make_sequence_classification(n_samples=1000, seq_len=20, vocab_size=50, n_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(1, vocab_size, (n_samples, seq_len))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if np.mean(X[i]) > vocab_size * 0.6:
            y[i] = 2
        elif np.mean(X[i]) > vocab_size * 0.4:
            y[i] = 1
        else:
            y[i] = 0
    return X, y


def make_sine_regression(n_samples=1000, seq_len=50, n_features=1, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, seq_len + 1)
    X = np.zeros((n_samples, seq_len, n_features))
    y = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        noise = rng.randn(seq_len + 1) * 0.05
        signal = np.sin(freq * t + phase) + noise
        X[i, :, 0] = signal[:seq_len]
        y[i, 0] = signal[seq_len]
    return X.astype(np.float32), y.astype(np.float32)


def make_spiral(n_samples_per_class=100, n_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    N = n_samples_per_class
    X = np.zeros((N * n_classes, 2), dtype=np.float32)
    y = np.zeros(N * n_classes, dtype=int)
    for c in range(n_classes):
        ix = range(N * c, N * (c + 1))
        r = np.linspace(0.0, 1, N)
        theta = np.linspace(c * 4, (c + 1) * 4, N) + rng.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[ix] = c
    return X, y


def make_xor(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 2).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y
