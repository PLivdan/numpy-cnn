import numpy as np
import pytest
from numpyml import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


class TestMultiHeadAttention:
    def test_forward_shape(self):
        mha = MultiHeadAttention(16, 4)
        mha.build((None, 10, 16))
        out = mha.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)

    def test_convergence(self):
        m = Model()
        m.add(MultiHeadAttention(16, 4))
        m.add(GlobalAvgPool1D())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 16), Adam(), 'he')
        X = np.random.randn(4, 10, 16)
        y = np.eye(5)[:4]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(10)]
        assert losses[-1] < losses[0]


class TestPositionalEncoding:
    def test_forward_shape(self):
        pe = PositionalEncoding(50)
        pe.build((None, 10, 16))
        out = pe.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)

    def test_adds_to_input(self):
        pe = PositionalEncoding(50)
        pe.build((None, 10, 16))
        x = np.zeros((1, 10, 16))
        out = pe.forward(x)
        assert not np.allclose(out, 0)

    def test_passthrough_backward(self):
        pe = PositionalEncoding(50)
        pe.build((None, 10, 16))
        pe.forward(np.random.randn(4, 10, 16))
        grads = np.ones((4, 10, 16))
        out = pe.backward(grads, 0.001)
        np.testing.assert_array_equal(out, grads)


class TestTransformerBlock:
    def test_pipeline(self):
        m = Model()
        m.add(PositionalEncoding(50))
        m.add(MultiHeadAttention(16, 4))
        m.add(LayerNorm())
        m.add(GlobalAvgPool1D())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 16), Adam(), 'he')
        X = np.random.randn(4, 10, 16)
        y = np.eye(5)[:4]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(5)]
        assert losses[-1] < losses[0]
