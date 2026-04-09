import numpy as np
import pytest
from numpycnn import *

OPTIMIZERS = [SGD, SGDmom, RMSprop, AdaGrad, AdaDelta, Adam, AdamW, NAG, AMSGrad, Yogi, AdaFactor]


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


@pytest.mark.parametrize("OptClass", OPTIMIZERS, ids=lambda c: c.__name__)
class TestOptimizer:
    def test_convergence(self, OptClass):
        m = Model()
        m.add(Conv2D(4, (3, 3), padding=0))
        m.add(Flatten())
        m.add(Dense(10, 'softmax'))
        m.compile((None, 8, 8, 1), OptClass(use_agc=True), 'he')
        X = np.random.randn(4, 8, 8, 1)
        y = np.eye(10)[:4]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(5)]
        assert not np.isnan(losses[-1])

    def test_no_nan(self, OptClass):
        m = Model()
        m.add(Dense(10, 'softmax'))
        m.compile((None, 20), OptClass(), 'he')
        X = np.random.randn(8, 20)
        y = np.eye(10)[np.random.randint(0, 10, 8)]
        loss = m.train_on_batch(X, y, categorical_crossentropy, 0.001)
        assert not np.isnan(loss)


class TestGradientClipping:
    def test_clip_norm(self):
        m = Model()
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8), Adam(clip_norm=1.0), 'he')
        loss = m.train_on_batch(np.random.randn(4, 8), np.eye(5)[:4], categorical_crossentropy, 0.01)
        assert not np.isnan(loss)

    def test_clip_value(self):
        m = Model()
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8), Adam(clip_value=0.5), 'he')
        loss = m.train_on_batch(np.random.randn(4, 8), np.eye(5)[:4], categorical_crossentropy, 0.01)
        assert not np.isnan(loss)
