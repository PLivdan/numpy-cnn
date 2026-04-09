import numpy as np
import pytest
from numpycnn import DataLoader, train_test_split, one_hot


class TestDataLoader:
    def test_length(self):
        dl = DataLoader(np.zeros((100, 5)), batch_size=32)
        assert len(dl) == 4

    def test_yields_all_data(self):
        X = np.arange(100).reshape(100, 1)
        dl = DataLoader(X, batch_size=32, shuffle=False)
        all_data = np.concatenate([b for b in dl], axis=0)
        assert all_data.shape == (100, 1)

    def test_two_arrays(self):
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 1)
        dl = DataLoader(X, y, batch_size=16)
        for xb, yb in dl:
            assert xb.shape[1] == 3
            assert yb.shape[1] == 1
            assert xb.shape[0] == yb.shape[0]

    def test_last_batch_smaller(self):
        dl = DataLoader(np.zeros((50, 3)), batch_size=32, shuffle=False)
        batches = list(dl)
        assert batches[0].shape[0] == 32
        assert batches[1].shape[0] == 18

    def test_shuffle(self):
        np.random.seed(42)
        X = np.arange(100).reshape(100, 1)
        dl = DataLoader(X, batch_size=100, shuffle=True)
        batch = next(iter(dl))
        assert not np.array_equal(batch, X)


class TestTrainTestSplit:
    def test_sizes(self):
        X = np.arange(100)
        a, b = train_test_split(X, test_size=0.2)
        assert len(a) == 80
        assert len(b) == 20

    def test_two_arrays(self):
        X = np.arange(100)
        y = np.arange(100)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
        assert len(X_tr) == 70
        assert len(y_te) == 30

    def test_deterministic_with_seed(self):
        X = np.arange(100)
        a1, b1 = train_test_split(X, test_size=0.2, random_state=42)
        a2, b2 = train_test_split(X, test_size=0.2, random_state=42)
        np.testing.assert_array_equal(a1, a2)


class TestOneHot:
    def test_shape(self):
        labels = np.array([0, 1, 2, 3])
        oh = one_hot(labels)
        assert oh.shape == (4, 4)

    def test_values(self):
        labels = np.array([0, 2])
        oh = one_hot(labels, num_classes=3)
        expected = np.array([[1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(oh, expected)
