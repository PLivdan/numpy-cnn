import numpy as np
import gzip
import os
import pickle
import urllib.request

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".numpycnn_data")


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
