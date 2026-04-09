import numpy as np
import pytest
from numpyml import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


class TestGroupNorm:
    def test_forward_backward(self):
        gn = GroupNorm(num_groups=4)
        gn.build((None, 8, 8, 16))
        gn.optimizer = Adam()
        gn.optimizer.init_params(gn)
        out = gn.forward(np.random.randn(2, 8, 8, 16))
        assert out.shape == (2, 8, 8, 16)
        grad = gn.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 8, 8, 16)

    def test_2d_input(self):
        gn = GroupNorm(num_groups=2)
        gn.build((None, 8))
        gn.optimizer = Adam()
        gn.optimizer.init_params(gn)
        out = gn.forward(np.random.randn(4, 8))
        assert out.shape == (4, 8)


class TestInstanceNorm:
    def test_forward_backward(self):
        inn = InstanceNorm()
        inn.build((None, 8, 8, 16))
        inn.optimizer = Adam()
        inn.optimizer.init_params(inn)
        out = inn.forward(np.random.randn(2, 8, 8, 16))
        assert out.shape == (2, 8, 8, 16)
        grad = inn.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 8, 8, 16)


class TestRMSNorm:
    def test_forward_backward(self):
        rn = RMSNorm()
        rn.build((None, 10, 16))
        rn.optimizer = Adam()
        rn.optimizer.init_params(rn)
        out = rn.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)
        grad = rn.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 10, 16)

    def test_no_nan(self):
        rn = RMSNorm()
        rn.build((None, 16))
        rn.optimizer = Adam()
        rn.optimizer.init_params(rn)
        out = rn.forward(np.random.randn(4, 16))
        assert not np.isnan(out).any()


class TestCausalAttention:
    def test_forward_shape(self):
        ca = CausalMultiHeadAttention(16, 4)
        ca.build((None, 10, 16))
        out = ca.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)

    def test_convergence(self):
        m = Model()
        m.add(CausalMultiHeadAttention(16, 4))
        m.add(GlobalAvgPool1D())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 16), Adam(), 'he')
        X = np.random.randn(4, 10, 16)
        y = np.eye(5)[:4]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(5)]
        assert losses[-1] < losses[0]

    def test_causal_mask(self):
        ca = CausalMultiHeadAttention(8, 2)
        ca.build((None, 5, 8))
        ca.forward(np.random.randn(1, 5, 8))
        weights = ca.attn_weights[0, 0]
        assert np.allclose(weights[0, 1:], 0, atol=1e-6)


class TestCrossAttention:
    def test_forward_shape(self):
        xa = CrossAttention(16, 4)
        xa.build((None, 10, 16))
        out = xa.forward(np.random.randn(4, 10, 16), np.random.randn(4, 20, 16))
        assert out.shape == (4, 10, 16)


class TestRotaryPositionalEncoding:
    def test_forward_backward(self):
        rope = RotaryPositionalEncoding(d_model=16, max_len=50)
        rope.build((None, 10, 16))
        out = rope.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)
        grad = rope.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 10, 16)

    def test_modifies_input(self):
        rope = RotaryPositionalEncoding(d_model=16, max_len=50)
        rope.build((None, 10, 16))
        x = np.random.randn(1, 10, 16)
        out = rope.forward(x.copy())
        assert not np.allclose(out, x)


class TestSpatialDropout:
    def test_drops_channels(self):
        sd = SpatialDropout(0.5)
        sd.build((None, 8, 8, 16))
        x = np.ones((2, 8, 8, 16))
        out = sd.forward(x, training=True)
        for c in range(16):
            channel = out[0, :, :, c]
            assert np.all(channel == 0) or np.all(channel != 0)

    def test_inference_passthrough(self):
        sd = SpatialDropout(0.5)
        sd.build((None, 8, 8, 16))
        x = np.ones((2, 8, 8, 16))
        out = sd.forward(x, training=False)
        np.testing.assert_array_equal(out, x)


class TestDropPath:
    def test_drops_samples(self):
        dp = DropPath(0.5)
        dp.build((None, 10, 16))
        np.random.seed(0)
        x = np.ones((100, 10, 16))
        out = dp.forward(x, training=True)
        dropped = np.all(out == 0, axis=(1, 2))
        assert dropped.any() and (~dropped).any()


class TestDilatedConv2D:
    def test_forward_shape(self):
        dc = DilatedConv2D(16, (3, 3), dilation=2, padding='same')
        dc.build((None, 16, 16, 3))
        out = dc.forward(np.random.randn(2, 16, 16, 3))
        assert out.shape == (2, 16, 16, 16)

    def test_convergence(self):
        m = Model()
        m.add(DilatedConv2D(8, (3, 3), dilation=2, padding='same'))
        m.add(Flatten())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8, 8, 1), Adam(), 'he')
        X = np.random.randn(4, 8, 8, 1)
        y = np.eye(5)[:4]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(5)]
        assert losses[-1] < losses[0]


class TestAdaptiveAvgPool2D:
    def test_output_size(self):
        ap = AdaptiveAvgPool2D((1, 1))
        ap.build((None, 7, 7, 64))
        out = ap.forward(np.random.randn(2, 7, 7, 64))
        assert out.shape == (2, 1, 1, 64)

    def test_non_square(self):
        ap = AdaptiveAvgPool2D((3, 3))
        ap.build((None, 14, 14, 32))
        out = ap.forward(np.random.randn(2, 14, 14, 32))
        assert out.shape == (2, 3, 3, 32)

    def test_backward_shape(self):
        ap = AdaptiveAvgPool2D((1, 1))
        ap.build((None, 7, 7, 64))
        out = ap.forward(np.random.randn(2, 7, 7, 64))
        grad = ap.backward(np.ones_like(out), 0.001)
        assert grad.shape == (2, 7, 7, 64)


class TestSEBlock:
    def test_forward_backward(self):
        m = Model()
        m.add(Conv2D(16, (3, 3), padding='same'))
        m.add(SEBlock(4))
        m.add(Flatten())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 8, 8, 1), Adam(), 'he')
        X = np.random.randn(2, 8, 8, 1)
        y = np.eye(5)[:2]
        loss = m.train_on_batch(X, y, categorical_crossentropy, 0.001)
        assert not np.isnan(loss)


class TestFeedForward:
    def test_forward_backward(self):
        ff = FeedForward(16, 64)
        ff.build((None, 10, 16))
        ff.optimizer = Adam()
        ff.optimizer.init_params(ff)
        out = ff.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)
        grad = ff.backward(np.ones_like(out), 0.001)
        assert grad.shape == (4, 10, 16)


class TestTransformerEncoderBlock:
    def test_convergence(self):
        m = Model()
        m.add(TransformerEncoderBlock(16, 4))
        m.add(GlobalAvgPool1D())
        m.add(Dense(5, 'softmax'))
        m.compile((None, 10, 16), Adam(), 'he')
        X = np.random.randn(4, 10, 16)
        y = np.eye(5)[:4]
        losses = [m.train_on_batch(X, y, categorical_crossentropy, 0.001) for _ in range(5)]
        assert losses[-1] < losses[0]


class TestTransformerDecoderBlock:
    def test_forward_shape(self):
        dec = TransformerDecoderBlock(16, 4)
        dec.build((None, 10, 16))
        dec.optimizer = Adam()
        for sub in [dec.self_attn, dec.norm1, dec.cross_attn, dec.norm2, dec.ff, dec.norm3]:
            sub.optimizer = dec.optimizer
            dec.optimizer.init_params(sub)
        out = dec.forward(np.random.randn(4, 10, 16))
        assert out.shape == (4, 10, 16)
