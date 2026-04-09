import numpy as np
import pytest
from numpycnn import Activation


ACTIVATIONS = ["relu", "leaky_relu", "elu", "selu", "gelu", "silu", "mish", "sigmoid", "tanh", "softmax"]


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


@pytest.mark.parametrize("act_name", ACTIVATIONS)
class TestActivation:
    def test_forward_shape(self, act_name):
        a = Activation(act_name)
        a.build((None, 10))
        out = a.forward(np.random.randn(4, 10))
        assert out.shape == (4, 10)

    def test_no_nan(self, act_name):
        a = Activation(act_name)
        a.build((None, 10))
        out = a.forward(np.random.randn(4, 10))
        assert not np.isnan(out).any()

    def test_backward_shape(self, act_name):
        a = Activation(act_name)
        a.build((None, 10))
        a.forward(np.random.randn(4, 10))
        grad = a.backward(np.ones((4, 10)), 0.001)
        assert grad.shape == (4, 10)
        assert not np.isnan(grad).any()

    def test_4d_input(self, act_name):
        a = Activation(act_name)
        a.build((None, 8, 8, 3))
        out = a.forward(np.random.randn(2, 8, 8, 3))
        assert out.shape == (2, 8, 8, 3)


class TestActivationProperties:
    def test_relu_zeros_negatives(self):
        a = Activation("relu")
        a.build((None, 5))
        out = a.forward(np.array([[-1, 0, 1, -2, 3]]))
        np.testing.assert_array_equal(out, [[0, 0, 1, 0, 3]])

    def test_sigmoid_range(self):
        a = Activation("sigmoid")
        a.build((None, 100))
        out = a.forward(np.random.randn(10, 100))
        assert out.min() >= 0
        assert out.max() <= 1

    def test_softmax_sums_to_one(self):
        a = Activation("softmax")
        a.build((None, 10))
        out = a.forward(np.random.randn(4, 10))
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_tanh_range(self):
        a = Activation("tanh")
        a.build((None, 100))
        out = a.forward(np.random.randn(10, 100))
        assert out.min() >= -1
        assert out.max() <= 1
