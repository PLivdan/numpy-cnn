import numpy as np
import pytest
from numpycnn import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


PREDS = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
TARGETS = np.array([[1, 0, 0], [0, 1, 0]])

LOSSES = [
    ("mse", mse_loss),
    ("cce", categorical_crossentropy),
    ("huber", huber_loss),
    ("bce", binary_crossentropy),
    ("focal", focal_loss),
    ("label_smooth", label_smoothing_crossentropy),
    ("kl_div", kl_divergence),
    ("hinge", hinge_loss),
]


@pytest.mark.parametrize("name,fn", LOSSES, ids=lambda x: x if isinstance(x, str) else "")
class TestLoss:
    def test_returns_scalar_loss(self, name, fn):
        loss, grads = fn(PREDS, TARGETS)
        assert np.isscalar(loss) or loss.ndim == 0

    def test_returns_correct_grad_shape(self, name, fn):
        loss, grads = fn(PREDS, TARGETS)
        assert grads.shape == PREDS.shape

    def test_no_nan(self, name, fn):
        loss, grads = fn(PREDS, TARGETS)
        assert not np.isnan(loss)
        assert not np.isnan(grads).any()

    def test_positive_loss(self, name, fn):
        loss, _ = fn(PREDS, TARGETS)
        assert loss >= 0


class TestLossProperties:
    def test_cce_perfect_prediction_low_loss(self):
        perfect = np.array([[0.99, 0.005, 0.005]])
        target = np.array([[1, 0, 0]])
        loss, _ = categorical_crossentropy(perfect, target)
        assert loss < 0.1

    def test_label_smoothing_higher_than_cce(self):
        loss_cce, _ = categorical_crossentropy(PREDS, TARGETS)
        loss_ls, _ = label_smoothing_crossentropy(PREDS, TARGETS, smoothing=0.1)
        assert loss_ls > loss_cce
