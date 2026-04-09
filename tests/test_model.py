import numpy as np
import pytest
import tempfile
import os
from numpycnn import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


def _build_small_model():
    m = Model()
    m.add(Conv2D(4, (3, 3), padding=0))
    m.add(Flatten())
    m.add(Dense(10, 'softmax'))
    m.compile((None, 8, 8, 1), Adam(), 'he')
    return m


class TestModelBasics:
    def test_compile_and_summary(self, capsys):
        m = _build_small_model()
        m.summary()
        captured = capsys.readouterr()
        assert "Total Parameters" in captured.out

    def test_forward(self):
        m = _build_small_model()
        out = m.forward(np.random.randn(2, 8, 8, 1))
        assert out.shape == (2, 10)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_predict(self):
        m = _build_small_model()
        out = m.predict(np.random.randn(2, 8, 8, 1))
        assert out.shape == (2, 10)

    def test_train_on_batch(self):
        m = _build_small_model()
        X = np.random.randn(4, 8, 8, 1)
        y = np.eye(10)[:4]
        loss = m.train_on_batch(X, y, categorical_crossentropy, 0.001)
        assert not np.isnan(loss)


class TestModelConvergence:
    def test_loss_decreases(self):
        m = _build_small_model()
        X = np.random.randn(8, 8, 8, 1)
        y = np.eye(10)[np.random.randint(0, 10, 8)]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(20)]
        assert losses[-1] < losses[0]


class TestSaveLoad:
    def test_roundtrip(self):
        m = _build_small_model()
        X = np.random.randn(2, 8, 8, 1)
        out1 = m.predict(X)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            m.save(path)
            m2 = Model.load(path)
            out2 = m2.predict(X)
            np.testing.assert_allclose(out1, out2)
        finally:
            os.unlink(path)

    def test_save_load_all_layer_types(self):
        m = Model()
        m.add(Conv2D(4, (3, 3), padding='same'))
        m.add(BatchNorm())
        m.add(Pooling2D((2, 2), 2))
        m.add(Flatten())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8, 8, 1), Adam(), 'he')
        X = np.random.randn(2, 8, 8, 1)
        out1 = m.predict(X)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            m.save(path)
            m2 = Model.load(path)
            np.testing.assert_allclose(out1, m2.predict(X))
        finally:
            os.unlink(path)


class TestFreeze:
    def test_freeze_stops_updates(self):
        m = Model()
        m.add(Dense(10, 'relu'))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8), Adam(), 'he')
        m.freeze([0])
        w_before = m.layers[0].params["W"].copy()
        X = np.random.randn(4, 8)
        y = np.eye(5)[:4]
        m.train_on_batch(X, y, categorical_crossentropy, 0.01)
        np.testing.assert_array_equal(m.layers[0].params["W"], w_before)

    def test_unfreeze_allows_updates(self):
        m = Model()
        m.add(Dense(10, 'relu'))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8), Adam(), 'he')
        m.freeze([0])
        m.unfreeze()
        w_before = m.layers[0].params["W"].copy()
        X = np.random.randn(4, 8)
        y = np.eye(5)[:4]
        m.train_on_batch(X, y, categorical_crossentropy, 0.01)
        assert not np.array_equal(m.layers[0].params["W"], w_before)

    def test_get_total_parameters(self):
        m = Model()
        m.add(Dense(10, 'relu'))
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8), Adam(), 'he')
        total, trainable = m.get_total_parameters()
        assert total == trainable
        m.freeze([0])
        total2, trainable2 = m.get_total_parameters()
        assert total2 == total
        assert trainable2 < total


class TestFit:
    def test_fit_returns_history(self):
        m = _build_small_model()
        X = np.random.randn(32, 8, 8, 1)
        y = np.eye(10)[np.random.randint(0, 10, 32)]
        Xv = np.random.randn(8, 8, 8, 1)
        yv = np.eye(10)[np.random.randint(0, 10, 8)]
        history = m.fit(X, y, Xv, yv, batch_size=16, epochs=2, loss_fn=categorical_crossentropy)
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
        assert len(history['train_accuracy']) == 2
        assert len(history['val_accuracy']) == 2

    def test_early_stopping(self):
        m = Model()
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8), Adam(), 'he')
        X = np.random.randn(32, 8)
        y = np.eye(5)[np.random.randint(0, 5, 32)]
        Xv = np.random.randn(8, 8)
        yv = np.eye(5)[np.random.randint(0, 5, 8)]
        es = EarlyStopping(patience=2)
        history = m.fit(X, y, Xv, yv, batch_size=16, epochs=50,
                        loss_fn=categorical_crossentropy, callbacks=[es])
        assert len(history['train_loss']) < 50
