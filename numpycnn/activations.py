import numpy as np
from .layers import BaseLayer


class Activation(BaseLayer):
    def __init__(self, activation='relu', alpha=0.01):
        super().__init__()
        self.activation = activation.lower()
        self.alpha = alpha
        self.layer_type = "Activation"

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if self.activation == 'relu':
            self.outputs = np.maximum(0, inputs)
        elif self.activation == 'leaky_relu':
            self.outputs = np.where(inputs > 0, inputs, self.alpha * inputs)
        elif self.activation == 'elu':
            self.outputs = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
        elif self.activation == 'selu':
            lam = 1.0507009873554804934193349852946
            alpha = 1.6732632423543772848170429916717
            self.outputs = lam * np.where(inputs > 0, inputs, alpha * (np.exp(inputs) - 1))
        elif self.activation == 'gelu':
            self._cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * inputs ** 3)))
            self.outputs = inputs * self._cdf
        elif self.activation == 'silu' or self.activation == 'swish':
            self._sigmoid = 1 / (1 + np.exp(-inputs))
            self.outputs = inputs * self._sigmoid
        elif self.activation == 'mish':
            self._sp = np.log1p(np.exp(inputs))
            self._tanh_sp = np.tanh(self._sp)
            self.outputs = inputs * self._tanh_sp
        elif self.activation == 'sigmoid':
            self.outputs = 1 / (1 + np.exp(-inputs))
        elif self.activation == 'tanh':
            self.outputs = np.tanh(inputs)
        elif self.activation == 'softmax':
            exps = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
            self.outputs = exps / np.sum(exps, axis=-1, keepdims=True)
        elif self.activation == 'linear' or self.activation == 'none':
            self.outputs = inputs
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self.outputs

    def backward(self, grads, learning_rate):
        x = self.inputs
        if self.activation == 'relu':
            return grads * (x > 0)
        elif self.activation == 'leaky_relu':
            return grads * np.where(x > 0, 1, self.alpha)
        elif self.activation == 'elu':
            return grads * np.where(x > 0, 1, self.outputs + self.alpha)
        elif self.activation == 'selu':
            lam = 1.0507009873554804934193349852946
            alpha = 1.6732632423543772848170429916717
            return grads * lam * np.where(x > 0, 1, alpha * np.exp(x))
        elif self.activation == 'gelu':
            c = np.sqrt(2 / np.pi)
            t = c * (x + 0.044715 * x ** 3)
            sech2 = 1 - np.tanh(t) ** 2
            return grads * (self._cdf + x * sech2 * c * (1 + 3 * 0.044715 * x ** 2))
        elif self.activation == 'silu' or self.activation == 'swish':
            return grads * (self._sigmoid + x * self._sigmoid * (1 - self._sigmoid))
        elif self.activation == 'mish':
            sigmoid_x = 1 / (1 + np.exp(-x))
            omega = 4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)
            delta = 2 * np.exp(x) + np.exp(2 * x) + 2
            return grads * (np.exp(x) * omega / (delta ** 2))
        elif self.activation == 'sigmoid':
            return grads * self.outputs * (1 - self.outputs)
        elif self.activation == 'tanh':
            return grads * (1 - self.outputs ** 2)
        elif self.activation == 'softmax':
            return grads
        else:
            return grads

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["Activation", self.output_shape[1:], 0, self.activation.upper()]
