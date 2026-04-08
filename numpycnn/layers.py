import numpy as np


class BaseLayer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.layer_type = "Base"
        self.params = {}
        self.optimizer = None
        self.initializer = 'xavier'

    def set_initializer(self, initializer):
        self.initializer = initializer

    def build(self, input_shape):
        self.input_shape = input_shape
        self._assert_input_shape(input_shape)
        if self.optimizer:
            self.optimizer.init_params(self)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError

    def update(self):
        pass

    def l2_regularization(self, l2_lambda):
        if "W" in self.params:
            return l2_lambda * np.sum(self.params["W"] ** 2) / 2
        return 0.0

    def get_num_parameters(self):
        return 0

    def summary(self):
        raise NotImplementedError

    def _assert_input_shape(self, input_shape):
        if len(self.input_shape[1:]) != len(input_shape[1:]):
            raise ValueError("Dimensions mismatch.")
        for expected_dim, actual_dim in zip(self.input_shape[1:], input_shape[1:]):
            if expected_dim is not None and expected_dim != actual_dim:
                raise ValueError(f"Expected {expected_dim}, but got {actual_dim}")


def im2col(inputs, kernel_size, stride, padding):
    N, H, W, C = inputs.shape
    KH, KW = kernel_size
    inputs_padded = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
    N, H_padded, W_padded, C = inputs_padded.shape
    OH = (H_padded - KH) // stride + 1
    OW = (W_padded - KW) // stride + 1
    col_matrix = np.zeros((KH * KW * C, N * OH * OW))
    col_idx = 0
    for i in range(0, H_padded - KH + 1, stride):
        for j in range(0, W_padded - KW + 1, stride):
            if col_idx + N > col_matrix.shape[1]:
                break
            patch = inputs_padded[:, i:i+KH, j:j+KW, :]
            col_matrix[:, col_idx:col_idx + N] = patch.reshape(N, -1).T
            col_idx += N
    return col_matrix


def col2im(col_matrix, input_shape, kernel_size, stride, padding):
    N, H, W, C = input_shape
    KH, KW = kernel_size
    dinputs_padded = np.zeros((N, H + 2 * padding, W + 2 * padding, C))
    col_idx = 0
    max_col_idx = col_matrix.shape[1] - N
    for i in range(0, H + 2 * padding - KH + 1, stride):
        for j in range(0, W + 2 * padding - KW + 1, stride):
            if col_idx > max_col_idx:
                break
            col_patch = col_matrix[:, col_idx:col_idx + N]
            col_patch = col_patch.reshape((KH, KW, C, N)).transpose(3, 0, 1, 2)
            dinputs_padded[:, i:i+KH, j:j+KW, :] += col_patch
            col_idx += N
    if padding > 0:
        dinputs = dinputs_padded[:, padding:-padding, padding:-padding, :]
    else:
        dinputs = dinputs_padded
    return dinputs


class Conv2D(BaseLayer):
    def __init__(self, filters, kernel_size, stride=1, padding=0, activation='relu', initializer="xavier"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.initializer = initializer
        self.layer_type = "Conv2D"
        self.params = {"W": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        C = input_shape[-1]
        KH, KW = self.kernel_size
        if self.padding == 'same':
            self.padding = ((input_shape[1] - 1) * self.stride + KH - input_shape[1]) // 2
        self.output_shape = self._calculate_output_shape(input_shape)
        if self.initializer == "xavier":
            self.params["W"] = np.random.randn(KH, KW, C, self.filters) * np.sqrt(2.0 / (KH * KW * C))
        elif self.initializer == "he":
            self.params["W"] = np.random.randn(KH, KW, C, self.filters) * np.sqrt(2.0 / (KH * KW * C))
        elif self.initializer == "random":
            self.params["W"] = np.random.randn(KH, KW, C, self.filters)
        else:
            raise ValueError("Invalid initializer. Choose from 'xavier', 'he', or 'random'.")
        self.params["b"] = np.zeros((1, 1, 1, self.filters))

    def _calculate_output_shape(self, input_shape):
        N, H, W, _ = input_shape
        KH, KW = self.kernel_size
        OH = (H + 2 * self.padding - KH) // self.stride + 1
        OW = (W + 2 * self.padding - KW) // self.stride + 1
        return (N, OH, OW, self.filters)

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        self.inputs = np.pad(inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        N, H, W, C = self.inputs.shape
        KH, KW = self.kernel_size
        _, OH, OW, _ = self.output_shape
        col_matrix = im2col(self.inputs, self.kernel_size, self.stride, 0)
        W_col = self.params["W"].reshape((KH * KW * C, self.filters))
        outputs = W_col.T @ col_matrix
        outputs = outputs + self.params["b"][0, 0, 0, :][:, np.newaxis]
        outputs = outputs.reshape((self.filters, OH, OW, N)).transpose(3, 1, 2, 0)
        if self.activation == 'relu':
            outputs = np.maximum(0, outputs)
        self.outputs = outputs
        return outputs

    def backward(self, grads, learning_rate):
        dparams = {"dW": np.zeros_like(self.params["W"]), "db": np.zeros_like(self.params["b"])}
        if self.activation == 'relu':
            grads[self.outputs <= 0] = 0
        N, H, W, C = self.inputs.shape
        KH, KW = self.kernel_size
        _, OH, OW, _ = self.output_shape
        grads_col = grads.transpose(3, 1, 2, 0).reshape((self.filters, -1))
        col_matrix = im2col(self.inputs, self.kernel_size, self.stride, 0)
        dparams["dW"] = grads_col @ col_matrix.T
        dparams["dW"] = dparams["dW"].reshape((self.filters, KH, KW, C)).transpose(1, 2, 3, 0)
        dparams["db"] = np.sum(grads, axis=(0, 1, 2)).reshape((1, 1, 1, self.filters))
        W_col = self.params["W"].reshape((self.filters, -1)).T
        dinputs_col = W_col @ grads_col
        dinputs = col2im(dinputs_col, self.inputs.shape, self.kernel_size, self.stride, self.padding)
        self.optimizer.update(self, dparams, learning_rate)
        if self.padding > 0:
            dinputs = dinputs[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return dinputs

    def get_num_parameters(self):
        return np.prod(self.params["W"].shape) + np.prod(self.params["b"].shape)

    def summary(self):
        return ["Convolution2D", self.output_shape[1:], self.get_num_parameters(), "relu" if self.activation == "relu" else self.activation]


class Pooling2D(BaseLayer):
    def __init__(self, pool_size=(2, 2), stride=2, mode='max'):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode.lower()
        self.layer_type = "Pooling2D"

    def build(self, input_shape):
        super().build(input_shape)
        batch_size, height, width, channels = input_shape
        pool_height, pool_width = self.pool_size
        self.output_shape = (
            batch_size,
            (height - pool_height) // self.stride + 1,
            (width - pool_width) // self.stride + 1,
            channels,
        )

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        self.inputs = inputs
        return self._pool(inputs)

    def _pool(self, inputs):
        batch_size, height, width, channels = inputs.shape
        pool_height, pool_width = self.pool_size
        outputs = np.zeros((batch_size, *self.output_shape[1:]))
        for h in range(self.output_shape[1]):
            for w in range(self.output_shape[2]):
                h_start, h_end = h * self.stride, h * self.stride + pool_height
                w_start, w_end = w * self.stride, w * self.stride + pool_width
                region = inputs[:, h_start:h_end, w_start:w_end, :]
                if self.mode == 'max':
                    outputs[:, h, w, :] = np.max(region, axis=(1, 2))
                elif self.mode == 'average':
                    outputs[:, h, w, :] = np.mean(region, axis=(1, 2))
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.mode}")
        return outputs

    def backward(self, grads, learning_rate):
        return self._pool_backward(grads)

    def _pool_backward(self, grads):
        batch_size, height, width, channels = self.inputs.shape
        pool_height, pool_width = self.pool_size
        dinputs = np.zeros_like(self.inputs)
        for h in range(self.output_shape[1]):
            for w in range(self.output_shape[2]):
                h_start, h_end = h * self.stride, h * self.stride + pool_height
                w_start, w_end = w * self.stride, w * self.stride + pool_width
                if self.mode == 'max':
                    region = self.inputs[:, h_start:h_end, w_start:w_end, :]
                    max_values = np.max(region, axis=(1, 2), keepdims=True)
                    mask = (region == max_values)
                    dinputs[:, h_start:h_end, w_start:w_end, :] += mask * grads[:, h, w, :][:, None, None, :]
                elif self.mode == 'average':
                    dA = grads[:, h, w, :][:, None, None, :] / (pool_height * pool_width)
                    dinputs[:, h_start:h_end, w_start:w_end, :] += dA
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.mode}")
        return dinputs

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["Pooling2D", self.output_shape[1:], 0, self.mode]


class GlobalAvgPool2D(BaseLayer):
    def __init__(self):
        super().__init__()
        self.layer_type = "GlobalAvgPool2D"

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = (input_shape[0], input_shape[-1])

    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.mean(inputs, axis=(1, 2))

    def backward(self, grads, learning_rate):
        N, H, W, C = self.inputs.shape
        return np.ones((N, H, W, C)) * grads[:, np.newaxis, np.newaxis, :] / (H * W)

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["GlobalAvgPool2D", self.output_shape[1:], 0, None]


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.layer_type = "Flatten"

    def build(self, input_shape):
        super().build(input_shape)
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], int(np.prod(input_shape[1:])))

    def forward(self, inputs, training=True):
        self.input_shape = inputs.shape
        batch_size = self.input_shape[0]
        self.outputs = inputs.reshape(batch_size, -1)
        return self.outputs

    def backward(self, grads, learning_rate):
        return grads.reshape(self.input_shape)

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["Flatten", self.output_shape[1:], 0, ""]


class Dense(BaseLayer):
    def __init__(self, units, activation="relu", initializer="xavier"):
        super().__init__()
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.layer_type = "Dense"
        self.params = {"W": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = np.prod(input_shape[1:])
        self.output_shape = (input_shape[0], self.units)
        if self.initializer == "xavier":
            self.params["W"] = np.random.randn(input_dim, self.units) * np.sqrt(2.0 / (input_dim + self.units))
        elif self.initializer == "he":
            self.params["W"] = np.random.randn(input_dim, self.units) * np.sqrt(2.0 / input_dim)
        elif self.initializer == "random":
            self.params["W"] = np.random.randn(input_dim, self.units)
        else:
            raise ValueError("Invalid initializer. Choose from 'xavier', 'he', or 'random'.")
        self.params["b"] = np.zeros((1, self.units))

    def forward(self, inputs, training=True):
        self.inputs = inputs
        flat_inputs = inputs.reshape(inputs.shape[0], -1)
        Z = np.dot(flat_inputs, self.params["W"]) + self.params["b"]
        if self.activation == "relu":
            self.A = np.maximum(0, Z)
        elif self.activation == "sigmoid":
            self.A = 1 / (1 + np.exp(-Z))
        elif self.activation == "tanh":
            self.A = np.tanh(Z)
        elif self.activation == "softmax":
            exps = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
            self.A = exps / np.sum(exps, axis=-1, keepdims=True)
        else:
            self.A = Z
        return self.A

    def backward(self, dA, learning_rate):
        dparams = {"dW": np.zeros_like(self.params["W"]), "db": np.zeros_like(self.params["b"])}
        batch_size = dA.shape[0]
        if self.activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[self.A <= 0] = 0
        elif self.activation == "sigmoid":
            dZ = dA * self.A * (1 - self.A)
        elif self.activation == "tanh":
            dZ = dA * (1 - np.square(self.A))
        elif self.activation == "softmax":
            dZ = dA
        else:
            dZ = dA
        flat_inputs = self.inputs.reshape(batch_size, -1)
        dparams["dW"] = np.dot(flat_inputs.T, dZ) / batch_size
        dparams["db"] = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dA_prev = np.dot(dZ, self.params["W"].T).reshape(self.inputs.shape)
        self.optimizer.update(self, dparams, learning_rate)
        return dA_prev

    def get_num_parameters(self):
        return np.prod(self.params["W"].shape) + np.prod(self.params["b"].shape)

    def summary(self):
        return ["Dense", self.output_shape[1:], self.get_num_parameters(), self.activation]


class BatchNorm(BaseLayer):
    def __init__(self, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_variance = None
        self.params = {"gamma": None, "beta": None}
        self.layer_type = "BatchNorm"

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape
        channels = input_shape[-1]
        self.params["gamma"] = np.ones(channels)
        self.params["beta"] = np.zeros(channels)
        self.running_mean = np.zeros(channels)
        self.running_variance = np.zeros(channels)

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        self.inputs = inputs
        axis = tuple(range(inputs.ndim - 1))
        mean = np.mean(inputs, axis=axis, keepdims=True)
        variance = np.var(inputs, axis=axis, keepdims=True)
        if training:
            self.normalized_inputs = (inputs - mean) / np.sqrt(variance + self.epsilon)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * np.squeeze(mean)
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * np.squeeze(variance)
        else:
            mean = self.running_mean.reshape((1,) * (inputs.ndim - 1) + (-1,))
            variance = self.running_variance.reshape((1,) * (inputs.ndim - 1) + (-1,))
            self.normalized_inputs = (inputs - mean) / np.sqrt(variance + self.epsilon)
        out = self.params["gamma"].reshape((1,) * (inputs.ndim - 1) + (-1,)) * self.normalized_inputs + self.params["beta"].reshape((1,) * (inputs.ndim - 1) + (-1,))
        return out

    def backward(self, grads, learning_rate):
        axis = tuple(range(self.inputs.ndim - 1))
        N = np.prod([self.inputs.shape[i] for i in axis])
        dparams = {}
        dparams["dgamma"] = np.sum(grads * self.normalized_inputs, axis=axis)
        dparams["dbeta"] = np.sum(grads, axis=axis)
        dnormalized = grads * self.params["gamma"].reshape((1,) * (grads.ndim - 1) + (-1,))
        rm = self.running_mean.reshape((1,) * (self.inputs.ndim - 1) + (-1,))
        rv = self.running_variance.reshape((1,) * (self.inputs.ndim - 1) + (-1,))
        dvariance = np.sum(dnormalized * (self.inputs - rm) * -0.5 * np.power(rv + self.epsilon, -1.5), axis=axis)
        dmean = np.sum(dnormalized * -1 / np.sqrt(rv + self.epsilon), axis=axis) + dvariance * np.sum(-2 * (self.inputs - rm), axis=axis) / N
        dA_prev = dnormalized / np.sqrt(rv + self.epsilon) + dvariance * 2 * (self.inputs - rm) / N + dmean / N
        self.optimizer.update(self, dparams, learning_rate)
        return dA_prev

    def get_num_parameters(self):
        return len(self.params["gamma"]) + len(self.params["beta"])

    def summary(self):
        return ["BatchNorm", self.output_shape[1:], self.get_num_parameters(), ""]


class LayerNorm(BaseLayer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.params = {"gamma": None, "beta": None}
        self.layer_type = "LayerNorm"

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape
        self.params["gamma"] = np.ones(input_shape[-1])
        self.params["beta"] = np.zeros(input_shape[-1])

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        self.inputs = inputs
        mean = np.mean(inputs, axis=-1, keepdims=True)
        variance = np.var(inputs, axis=-1, keepdims=True)
        self.normalized_inputs = (inputs - mean) / np.sqrt(variance + self.epsilon)
        return self.params["gamma"] * self.normalized_inputs + self.params["beta"]

    def backward(self, grads, learning_rate):
        dparams = {}
        axis_to_sum = tuple(range(len(grads.shape) - 1))
        dparams["dgamma"] = np.sum(grads * self.normalized_inputs, axis=axis_to_sum)
        dparams["dbeta"] = np.sum(grads, axis=axis_to_sum)
        dnormalized = grads * self.params["gamma"]
        N = dnormalized.shape[-1]
        mean = np.mean(self.inputs, axis=-1, keepdims=True)
        variance = np.var(self.inputs, axis=-1, keepdims=True)
        dvariance = np.sum(dnormalized * (self.inputs - mean) * -0.5 * np.power(variance + self.epsilon, -1.5), axis=-1, keepdims=True)
        dmean = np.sum(dnormalized * -1 / np.sqrt(variance + self.epsilon), axis=-1, keepdims=True) + dvariance * np.sum(-2 * (self.inputs - mean), axis=-1, keepdims=True) / N
        dA_prev = dnormalized / np.sqrt(variance + self.epsilon) + dvariance * 2 * (self.inputs - mean) / N + dmean / N
        self.optimizer.update(self, dparams, learning_rate)
        return dA_prev

    def get_num_parameters(self):
        return len(self.params["gamma"]) + len(self.params["beta"])

    def summary(self):
        return ["LayerNorm", self.output_shape[1:], self.get_num_parameters(), ""]


class Dropout(BaseLayer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.layer_type = "Dropout"
        self.mask = None

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape

    def forward(self, inputs, training=True):
        self._assert_input_shape(inputs.shape)
        if training:
            self.mask = (np.random.rand(*inputs.shape) >= self.rate) / (1 - self.rate)
            return inputs * self.mask
        return inputs

    def backward(self, grads, learning_rate):
        return grads * self.mask

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["Dropout", self.output_shape[1:], 0, ""]


class SkipConnection(BaseLayer):
    def __init__(self, skip_from, operation='add'):
        super().__init__()
        self.skip_from = skip_from
        self.operation = operation
        self.layer_type = "SkipConnection"
        self.alpha = 1.0

    def build(self, input_shape):
        self.output_shape = input_shape
        self.skip_input_shape = None
        self.W = None
        self.resized_skip_input = None

    def forward(self, inputs, skip_input, training=True):
        self.inputs = inputs
        if inputs.shape != skip_input.shape:
            self.skip_input_shape = skip_input.shape
            target_channels = inputs.shape[-1]
            if self.W is None:
                self.W = np.random.randn(1, 1, skip_input.shape[-1], target_channels)
            col_skip_input = im2col(skip_input, (1, 1), 1, 0)
            temp_W = self.W.reshape(-1, target_channels)
            temp_output = temp_W.dot(col_skip_input)
            skip_input = col2im(temp_output, inputs.shape, (1, 1), 1, 0)
        self.resized_skip_input = skip_input
        if self.operation == 'add':
            return self.alpha * inputs + (1 - self.alpha) * skip_input
        elif self.operation == 'multiply':
            return (self.alpha * inputs) * ((1 - self.alpha) * skip_input + 1)
        elif self.operation == 'concat':
            return np.concatenate([self.alpha * inputs, (1 - self.alpha) * skip_input], axis=-1)
        raise ValueError(f"Unsupported operation: {self.operation}")

    def backward(self, grads, learning_rate):
        d_alpha = np.sum(grads * (self.inputs - self.resized_skip_input))
        self.alpha -= learning_rate * d_alpha
        if self.skip_input_shape is not None:
            col_grads = im2col(grads, (1, 1), 1, 0)
            dW = col_grads.dot(grads.reshape(-1, self.skip_input_shape[-1])).reshape(self.W.shape)
            self.W -= learning_rate * dW
        return grads

    def summary(self):
        return ["SkipConnection", self.output_shape[1:], 1, f"Skip from {self.skip_from}"]
