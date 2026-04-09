import numpy as np
import pytest
from numpycnn import accuracy, precision, recall, f1_score, confusion_matrix, top_k_accuracy


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


Y_TRUE = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
Y_PRED = np.array([0, 1, 2, 3, 0, 0, 1, 2, 3, 4])


class TestAccuracy:
    def test_basic(self):
        assert accuracy(Y_TRUE, Y_PRED) == 0.9

    def test_one_hot(self):
        yt = np.eye(5)[Y_TRUE]
        yp = np.eye(5)[Y_PRED]
        assert accuracy(yt, yp) == 0.9

    def test_perfect(self):
        assert accuracy(Y_TRUE, Y_TRUE) == 1.0


class TestConfusionMatrix:
    def test_shape(self):
        cm = confusion_matrix(Y_TRUE, Y_PRED)
        assert cm.shape == (5, 5)

    def test_diagonal_for_perfect(self):
        cm = confusion_matrix(Y_TRUE, Y_TRUE)
        assert np.all(cm == np.diag(np.diag(cm)))

    def test_sum_equals_samples(self):
        cm = confusion_matrix(Y_TRUE, Y_PRED)
        assert cm.sum() == len(Y_TRUE)


class TestPrecisionRecallF1:
    def test_precision_range(self):
        p = precision(Y_TRUE, Y_PRED, average='macro')
        assert 0 <= p <= 1

    def test_recall_range(self):
        r = recall(Y_TRUE, Y_PRED, average='macro')
        assert 0 <= r <= 1

    def test_f1_range(self):
        f = f1_score(Y_TRUE, Y_PRED, average='macro')
        assert 0 <= f <= 1

    def test_perfect_scores(self):
        assert precision(Y_TRUE, Y_TRUE) == pytest.approx(1.0, abs=1e-6)
        assert recall(Y_TRUE, Y_TRUE) == pytest.approx(1.0, abs=1e-6)
        assert f1_score(Y_TRUE, Y_TRUE) == pytest.approx(1.0, abs=1e-6)

    def test_per_class(self):
        p = precision(Y_TRUE, Y_PRED, average=None)
        assert len(p) == 5


class TestTopKAccuracy:
    def test_top1(self):
        probs = np.eye(5)[Y_PRED]
        assert top_k_accuracy(Y_TRUE, probs, k=1) == accuracy(Y_TRUE, Y_PRED)

    def test_top5_is_one(self):
        probs = np.random.randn(10, 5)
        assert top_k_accuracy(Y_TRUE, probs, k=5) == 1.0
