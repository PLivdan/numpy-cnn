import numpy as np
from .layers import BaseLayer


class SpatialDropout(BaseLayer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.layer_type = "SpatialDropout"
        self.mask = None

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        if training:
            if inputs.ndim == 4:
                N, H, W, C = inputs.shape
                self.mask = (np.random.rand(N, 1, 1, C) >= self.rate) / (1 - self.rate)
            elif inputs.ndim == 3:
                N, L, C = inputs.shape
                self.mask = (np.random.rand(N, 1, C) >= self.rate) / (1 - self.rate)
            else:
                self.mask = (np.random.rand(*inputs.shape) >= self.rate) / (1 - self.rate)
            return inputs * self.mask
        return inputs

    def backward(self, grads, learning_rate):
        return grads * self.mask

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["SpatialDropout", self.output_shape[1:], 0, f"rate={self.rate}"]


class DropPath(BaseLayer):
    def __init__(self, rate=0.1):
        super().__init__()
        self.rate = rate
        self.layer_type = "DropPath"
        self.mask = None

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        if training and self.rate > 0:
            keep = 1 - self.rate
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
            self.mask = (np.random.rand(*shape) < keep) / keep
            return inputs * self.mask
        self.mask = None
        return inputs

    def backward(self, grads, learning_rate):
        if self.mask is not None:
            return grads * self.mask
        return grads

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["DropPath", self.output_shape[1:], 0, f"rate={self.rate}"]
