import numpy as np
import pytest
from numpycnn import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


X = None
Y = None


@pytest.fixture(autouse=True)
def data():
    global X, Y
    X = np.random.randn(4, 10, 8)
    Y = np.eye(5)[:4]


class TestRNN:
    def test_forward_shape(self):
        r = RNN(32)
        r.build((None, 10, 8))
        assert r.forward(X).shape == (4, 32)

    def test_return_sequences(self):
        r = RNN(32, return_sequences=True)
        r.build((None, 10, 8))
        assert r.forward(X).shape == (4, 10, 32)

    def test_convergence(self):
        m = Model()
        m.add(RNN(32))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 8), Adam(), 'he')
        losses = [m.train_on_batch(X, Y, categorical_crossentropy, 0.01) for _ in range(10)]
        assert losses[-1] < losses[0]


class TestLSTM:
    def test_forward_shape(self):
        l = LSTM(32)
        l.build((None, 10, 8))
        assert l.forward(X).shape == (4, 32)

    def test_return_sequences(self):
        l = LSTM(32, return_sequences=True)
        l.build((None, 10, 8))
        assert l.forward(X).shape == (4, 10, 32)

    def test_convergence(self):
        m = Model()
        m.add(LSTM(32))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 8), Adam(), 'he')
        losses = [m.train_on_batch(X, Y, categorical_crossentropy, 0.01) for _ in range(10)]
        assert losses[-1] < losses[0]

    def test_stacked(self):
        m = Model()
        m.add(LSTM(32, return_sequences=True))
        m.add(LSTM(16))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 8), Adam(), 'he')
        loss = m.train_on_batch(X, Y, categorical_crossentropy, 0.01)
        assert not np.isnan(loss)


class TestGRU:
    def test_forward_shape(self):
        g = GRU(32)
        g.build((None, 10, 8))
        assert g.forward(X).shape == (4, 32)

    def test_convergence(self):
        m = Model()
        m.add(GRU(32))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 8), Adam(), 'he')
        losses = [m.train_on_batch(X, Y, categorical_crossentropy, 0.01) for _ in range(10)]
        assert losses[-1] < losses[0]


class TestBidirectional:
    def test_lstm_concat(self):
        m = Model()
        m.add(Bidirectional(LSTM(16)))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 8), Adam(), 'he')
        out = m.forward(X)
        assert out.shape == (4, 5)

    def test_gru_seq(self):
        m = Model()
        m.add(Bidirectional(GRU(16, return_sequences=True)))
        m.add(GlobalAvgPool1D())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 8), Adam(), 'he')
        losses = [m.train_on_batch(X, Y, categorical_crossentropy, 0.01) for _ in range(5)]
        assert losses[-1] < losses[0]


class TestEmbeddingRNN:
    def test_pipeline(self):
        m = Model()
        m.add(Embedding(100, 16))
        m.add(LSTM(32))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10), Adam(), 'he')
        Xt = np.random.randint(0, 100, (4, 10))
        losses = [m.train_on_batch(Xt, Y, categorical_crossentropy, 0.01) for _ in range(5)]
        assert losses[-1] < losses[0]
