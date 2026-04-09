import numpy as np
import pytest
from numpycnn import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


class TestConv2D:
    def test_build_shapes(self):
        c = Conv2D(8, (3, 3), padding=1)
        c.build((None, 28, 28, 1))
        assert c.output_shape == (None, 28, 28, 8)
        assert c.params["W"].shape == (3, 3, 1, 8)
        assert c.params["b"].shape == (1, 1, 1, 8)

    def test_same_padding(self):
        c = Conv2D(16, (5, 5), padding='same')
        c.build((None, 28, 28, 3))
        assert c.output_shape == (None, 28, 28, 16)

    def test_forward_shape(self):
        c = Conv2D(8, (3, 3), padding=1)
        c.build((None, 28, 28, 1))
        out = c.forward(np.random.randn(4, 28, 28, 1))
        assert out.shape == (4, 28, 28, 8)

    def test_no_nan(self):
        c = Conv2D(8, (3, 3), padding=1)
        c.build((None, 28, 28, 1))
        c.optimizer = Adam()
        c.optimizer.init_params(c)
        out = c.forward(np.random.randn(4, 28, 28, 1))
        grad = c.backward(np.ones_like(out), 0.001)
        assert not np.isnan(out).any()
        assert not np.isnan(grad).any()

    def test_stride(self):
        c = Conv2D(8, (3, 3), stride=2, padding=0)
        c.build((None, 28, 28, 1))
        assert c.output_shape == (None, 13, 13, 8)


class TestConv1D:
    def test_forward_backward(self):
        c = Conv1D(16, 3, padding=1)
        c.build((None, 20, 8))
        c.optimizer = Adam()
        c.optimizer.init_params(c)
        out = c.forward(np.random.randn(4, 20, 8))
        assert out.shape == (4, 20, 16)
        grad = c.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 20, 8)


class TestConvTranspose2D:
    def test_output_shape(self):
        ct = ConvTranspose2D(8, (3, 3), stride=2)
        ct.build((None, 4, 4, 16))
        assert ct.output_shape == (None, 9, 9, 8)

    def test_forward_backward(self):
        ct = ConvTranspose2D(8, (3, 3), stride=2)
        ct.build((None, 4, 4, 16))
        ct.optimizer = Adam()
        ct.optimizer.init_params(ct)
        out = ct.forward(np.random.randn(2, 4, 4, 16))
        grad = ct.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 4, 4, 16)


class TestSeparableConv2D:
    def test_forward_backward(self):
        s = SeparableConv2D(16, (3, 3), padding='same')
        s.build((None, 8, 8, 3))
        s.optimizer = Adam()
        s.optimizer.init_params(s)
        out = s.forward(np.random.randn(2, 8, 8, 3))
        assert out.shape == (2, 8, 8, 16)
        grad = s.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 8, 8, 3)


class TestPooling2D:
    @pytest.mark.parametrize("mode", ["max", "average"])
    def test_forward_shape(self, mode):
        p = Pooling2D((2, 2), 2, mode)
        p.build((None, 28, 28, 8))
        out = p.forward(np.random.randn(4, 28, 28, 8))
        assert out.shape == (4, 14, 14, 8)

    @pytest.mark.parametrize("mode", ["max", "average"])
    def test_backward_shape(self, mode):
        p = Pooling2D((2, 2), 2, mode)
        p.build((None, 28, 28, 8))
        out = p.forward(np.random.randn(4, 28, 28, 8))
        grad = p.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 28, 28, 8)


class TestPooling1D:
    def test_forward_backward(self):
        p = Pooling1D(2, 2)
        p.build((None, 20, 8))
        out = p.forward(np.random.randn(4, 20, 8))
        assert out.shape == (4, 10, 8)
        grad = p.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 20, 8)


class TestGlobalAvgPool:
    def test_2d(self):
        g = GlobalAvgPool2D()
        g.build((None, 7, 7, 64))
        out = g.forward(np.random.randn(2, 7, 7, 64))
        assert out.shape == (2, 64)

    def test_1d(self):
        g = GlobalAvgPool1D()
        g.build((None, 10, 32))
        out = g.forward(np.random.randn(2, 10, 32))
        assert out.shape == (2, 32)


class TestDense:
    @pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh", "softmax", "linear"])
    def test_activations(self, act):
        d = Dense(10, activation=act)
        d.build((None, 20))
        d.optimizer = Adam()
        d.optimizer.init_params(d)
        out = d.forward(np.random.randn(4, 20))
        assert out.shape == (4, 10)
        assert not np.isnan(out).any()
        grad = d.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 20)

    def test_softmax_sums_to_one(self):
        d = Dense(10, activation="softmax")
        d.build((None, 20))
        out = d.forward(np.random.randn(4, 20))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-6)


class TestNormalization:
    def test_batchnorm_forward_backward(self):
        bn = BatchNorm()
        bn.build((None, 8, 8, 16))
        bn.optimizer = Adam()
        bn.optimizer.init_params(bn)
        out = bn.forward(np.random.randn(4, 8, 8, 16))
        assert out.shape == (4, 8, 8, 16)
        grad = bn.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 8, 8, 16)

    def test_layernorm_forward_backward(self):
        ln = LayerNorm()
        ln.build((None, 8, 8, 16))
        ln.optimizer = Adam()
        ln.optimizer.init_params(ln)
        out = ln.forward(np.random.randn(4, 8, 8, 16))
        grad = ln.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 8, 8, 16)

    def test_batchnorm_inference_mode(self):
        bn = BatchNorm()
        bn.build((None, 10))
        bn.optimizer = Adam()
        bn.optimizer.init_params(bn)
        bn.forward(np.random.randn(8, 10), training=True)
        out = bn.forward(np.random.randn(4, 10), training=False)
        assert out.shape == (4, 10)


class TestDropout:
    def test_training_mode(self):
        d = Dropout(0.5)
        d.build((None, 100))
        out = d.forward(np.ones((4, 100)), training=True)
        assert (out == 0).any()

    def test_inference_mode(self):
        d = Dropout(0.5)
        d.build((None, 100))
        x = np.ones((4, 100))
        out = d.forward(x, training=False)
        np.testing.assert_array_equal(out, x)


class TestReshapeFlatten:
    def test_flatten(self):
        f = Flatten()
        f.build((None, 7, 7, 64))
        assert f.output_shape == (None, 3136)
        out = f.forward(np.random.randn(2, 7, 7, 64))
        assert out.shape == (2, 3136)

    def test_reshape(self):
        r = Reshape((7, 7, 1))
        r.build((None, 49))
        out = r.forward(np.random.randn(2, 49))
        assert out.shape == (2, 7, 7, 1)
        grad = r.backward(np.ones((2, 7, 7, 1)), 0.0)
        assert grad.shape == (2, 49)


class TestUpsample:
    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    def test_forward_backward(self, mode):
        u = Upsample2D(2, mode)
        u.build((None, 4, 4, 3))
        out = u.forward(np.random.randn(2, 4, 4, 3))
        assert out.shape == (2, 8, 8, 3)
        grad = u.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 4, 4, 3)


class TestZeroPadding:
    def test_forward_backward(self):
        z = ZeroPadding2D((2, 2))
        z.build((None, 4, 4, 3))
        out = z.forward(np.random.randn(2, 4, 4, 3))
        assert out.shape == (2, 8, 8, 3)
        grad = z.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 4, 4, 3)


class TestEmbedding:
    def test_forward_backward(self):
        e = Embedding(100, 16)
        e.build((None, 10))
        e.optimizer = Adam()
        e.optimizer.init_params(e)
        out = e.forward(np.random.randint(0, 100, (4, 10)))
        assert out.shape == (4, 10, 16)
        e.backward(np.ones_like(out), 0.001)


class TestSkipConnection:
    def test_add(self):
        model = Model()
        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(SkipConnection(skip_from=0))
        model.add(Flatten())
        model.add(Dense(5, activation='softmax'))
        model.compile((None, 8, 8, 1), Adam(), 'he')
        out = model.forward(np.random.randn(2, 8, 8, 1))
        assert out.shape == (2, 5)


class TestIm2Col:
    def test_roundtrip(self):
        x = np.random.randn(2, 8, 8, 3)
        col = im2col(x, (3, 3), 1, 0)
        reconstructed = col2im(col, x.shape, (3, 3), 1, 0)
        assert reconstructed.shape == x.shape

    def test_roundtrip_1d(self):
        x = np.random.randn(2, 20, 4)
        col = im2col_1d(x, 3, 1, 0)
        reconstructed = col2im_1d(col, x.shape, 3, 1, 0)
        assert reconstructed.shape == x.shape
