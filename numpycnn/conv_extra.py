import numpy as np
from .layers import BaseLayer, im2col, col2im


class DilatedConv2D(BaseLayer):
    def __init__(self, filters, kernel_size, dilation=2, padding=0, activation='relu', initializer="xavier"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.dilation = dilation
        self.padding = padding
        self.activation = activation
        self.initializer = initializer
        self.layer_type = "DilatedConv2D"
        self.params = {"W": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, H, W, C = input_shape
        KH, KW = self.kernel_size
        d = self.dilation
        eKH = KH + (KH - 1) * (d - 1)
        eKW = KW + (KW - 1) * (d - 1)
        if self.padding == 'same':
            self.padding = (eKH - 1) // 2
        oH = (H + 2 * self.padding - eKH) + 1
        oW = (W + 2 * self.padding - eKW) + 1
        self.output_shape = (N, oH, oW, self.filters)
        fan_in = KH * KW * C
        if self.initializer == "he":
            self.params["W"] = np.random.randn(KH, KW, C, self.filters) * np.sqrt(2.0 / fan_in)
        else:
            self.params["W"] = np.random.randn(KH, KW, C, self.filters) * np.sqrt(2.0 / (fan_in + KH * KW * self.filters))
        self.params["b"] = np.zeros((1, 1, 1, self.filters))

    def _dilate_kernel(self, W):
        KH, KW, C, F = W.shape
        d = self.dilation
        eKH = KH + (KH - 1) * (d - 1)
        eKW = KW + (KW - 1) * (d - 1)
        W_dilated = np.zeros((eKH, eKW, C, F))
        W_dilated[::d, ::d, :, :] = W
        return W_dilated

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        self.inputs = inputs
        if self.padding > 0:
            self.inputs_padded = np.pad(inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        else:
            self.inputs_padded = inputs
        N, H, W, C = self.inputs_padded.shape
        _, oH, oW, _ = self.output_shape
        W_dilated = self._dilate_kernel(self.params["W"])
        eKH, eKW = W_dilated.shape[:2]
        self._col = im2col(self.inputs_padded, (eKH, eKW), 1, 0)
        W_col = W_dilated.reshape(eKH * eKW * C, self.filters)
        outputs = W_col.T @ self._col
        outputs = outputs + self.params["b"][0, 0, 0, :][:, np.newaxis]
        outputs = outputs.reshape(self.filters, oH, oW, N).transpose(3, 1, 2, 0)
        if self.activation == 'relu':
            outputs = np.maximum(0, outputs)
        self.outputs = outputs
        return outputs

    def backward(self, grads, learning_rate):
        if self.activation == 'relu':
            grads = grads * (self.outputs > 0)
        N, H, W, C = self.inputs_padded.shape
        KH, KW = self.kernel_size
        d = self.dilation
        W_dilated = self._dilate_kernel(self.params["W"])
        eKH, eKW = W_dilated.shape[:2]
        _, oH, oW, _ = self.output_shape
        grads_col = grads.transpose(3, 1, 2, 0).reshape(self.filters, -1)
        dW_dilated_col = grads_col @ self._col.T
        dW_dilated = dW_dilated_col.reshape(self.filters, eKH, eKW, C).transpose(1, 2, 3, 0)
        dparams = {}
        dparams["dW"] = dW_dilated[::d, ::d, :, :]
        dparams["db"] = np.sum(grads, axis=(0, 1, 2)).reshape(1, 1, 1, self.filters)
        W_col = W_dilated.reshape(self.filters, -1).T
        dinputs_col = W_col @ grads_col
        dinputs = col2im(dinputs_col, self.inputs_padded.shape, (eKH, eKW), 1, 0)
        if self.padding > 0:
            dinputs = dinputs[:, self.padding:-self.padding, self.padding:-self.padding, :]
        if self.trainable:
            self.optimizer.update(self, dparams, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return np.prod(self.params["W"].shape) + np.prod(self.params["b"].shape)

    def summary(self):
        return ["DilatedConv2D", self.output_shape[1:], self.get_num_parameters(),
                f"d={self.dilation}"]


class AdaptiveAvgPool2D(BaseLayer):
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.layer_type = "AdaptiveAvgPool2D"

    def build(self, input_shape):
        super().build(input_shape)
        N, H, W, C = input_shape
        self.output_shape = (N, self.output_size[0], self.output_size[1], C)

    def _get_regions(self, dim_in, dim_out):
        starts = (np.arange(dim_out) * dim_in / dim_out).astype(int)
        ends = ((np.arange(dim_out) + 1) * dim_in / dim_out).astype(int)
        return starts, ends

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, H, W, C = inputs.shape
        oH, oW = self.output_size
        h_starts, h_ends = self._get_regions(H, oH)
        w_starts, w_ends = self._get_regions(W, oW)
        outputs = np.zeros((N, oH, oW, C))
        for i in range(oH):
            for j in range(oW):
                outputs[:, i, j, :] = np.mean(inputs[:, h_starts[i]:h_ends[i], w_starts[j]:w_ends[j], :], axis=(1, 2))
        return outputs

    def backward(self, grads, learning_rate):
        N, H, W, C = self.inputs.shape
        oH, oW = self.output_size
        h_starts, h_ends = self._get_regions(H, oH)
        w_starts, w_ends = self._get_regions(W, oW)
        dinputs = np.zeros_like(self.inputs)
        for i in range(oH):
            for j in range(oW):
                h_size = h_ends[i] - h_starts[i]
                w_size = w_ends[j] - w_starts[j]
                dinputs[:, h_starts[i]:h_ends[i], w_starts[j]:w_ends[j], :] += grads[:, i, j, :][:, None, None, :] / (h_size * w_size)
        return dinputs

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["AdaptiveAvgPool2D", self.output_shape[1:], 0, f"{self.output_size}"]
