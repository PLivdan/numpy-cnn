import numpy as np


class DataLoader:
    def __init__(self, *arrays, batch_size=32, shuffle=True):
        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = arrays[0].shape[0]

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, self.n, self.batch_size):
            idx = indices[i:i + self.batch_size]
            if len(self.arrays) == 1:
                yield self.arrays[0][idx]
            else:
                yield tuple(a[idx] for a in self.arrays)


def train_test_split(*arrays, test_size=0.2, shuffle=True, random_state=None):
    n = arrays[0].shape[0]
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    split = int(n * (1 - test_size))
    result = []
    for a in arrays:
        result.append(a[indices[:split]])
        result.append(a[indices[split:]])
    return result


def one_hot(labels, num_classes=None):
    if num_classes is None:
        num_classes = labels.max() + 1
    return np.eye(num_classes)[labels]
