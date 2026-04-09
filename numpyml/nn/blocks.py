import numpy as np
from .layers import BaseLayer, Dense, LayerNorm, Dropout


class SEBlock(BaseLayer):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.layer_type = "SEBlock"
        self.params = {"W1": None, "b1": None, "W2": None, "b2": None}

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape
        C = input_shape[-1]
        mid = max(C // self.reduction, 1)
        self.params["W1"] = np.random.randn(C, mid) * np.sqrt(2.0 / C)
        self.params["b1"] = np.zeros((1, mid))
        self.params["W2"] = np.random.randn(mid, C) * np.sqrt(2.0 / mid)
        self.params["b2"] = np.zeros((1, C))
        self._mid = mid

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if inputs.ndim == 4:
            self._squeeze = np.mean(inputs, axis=(1, 2))
        else:
            self._squeeze = inputs
        self._fc1 = np.maximum(0, self._squeeze @ self.params["W1"] + self.params["b1"])
        z = self._fc1 @ self.params["W2"] + self.params["b2"]
        self._scale = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        if inputs.ndim == 4:
            return inputs * self._scale[:, None, None, :]
        return inputs * self._scale

    def backward(self, grads, learning_rate):
        if self.inputs.ndim == 4:
            dscale = np.sum(grads * self.inputs, axis=(1, 2))
            dinputs = grads * self._scale[:, None, None, :]
        else:
            dscale = grads * self.inputs
            dinputs = grads * self._scale
        dsigmoid = dscale * self._scale * (1 - self._scale)
        dparams = {}
        dparams["dW2"] = self._fc1.T @ dsigmoid / grads.shape[0]
        dparams["db2"] = np.sum(dsigmoid, axis=0, keepdims=True) / grads.shape[0]
        dfc1 = dsigmoid @ self.params["W2"].T
        dfc1 = dfc1 * (self._fc1 > 0)
        dparams["dW1"] = self._squeeze.T @ dfc1 / grads.shape[0]
        dparams["db1"] = np.sum(dfc1, axis=0, keepdims=True) / grads.shape[0]
        dsqueeze = dfc1 @ self.params["W1"].T
        if self.inputs.ndim == 4:
            N, H, W, C = self.inputs.shape
            dinputs += dsqueeze[:, None, None, :] / (H * W)
        else:
            dinputs += dsqueeze
        if self.trainable:
            self.optimizer.update(self, dparams, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return sum(np.prod(v.shape) for v in self.params.values())

    def summary(self):
        return ["SEBlock", self.output_shape[1:], self.get_num_parameters(),
                f"r={self.reduction}"]


class FeedForward(BaseLayer):
    def __init__(self, d_model, d_ff=None, activation='gelu', dropout_rate=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or d_model * 4
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer_type = "FeedForward"
        self.params = {"W1": None, "b1": None, "W2": None, "b2": None}

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape
        d, ff = self.d_model, self.d_ff
        self.params["W1"] = np.random.randn(d, ff) * np.sqrt(2.0 / d)
        self.params["b1"] = np.zeros((1, 1, ff)) if len(input_shape) == 3 else np.zeros((1, ff))
        self.params["W2"] = np.random.randn(ff, d) * np.sqrt(2.0 / ff)
        self.params["b2"] = np.zeros((1, 1, d)) if len(input_shape) == 3 else np.zeros((1, d))

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        return x

    def _activate_backward(self, x, dout):
        if self.activation == 'relu':
            return dout * (x > 0)
        elif self.activation == 'gelu':
            c = np.sqrt(2 / np.pi)
            t = c * (x + 0.044715 * x ** 3)
            cdf = 0.5 * (1 + np.tanh(t))
            sech2 = 1 - np.tanh(t) ** 2
            return dout * (cdf + x * sech2 * c * (1 + 3 * 0.044715 * x ** 2))
        return dout

    def forward(self, inputs, training=True):
        self.inputs = inputs
        self._z1 = inputs @ self.params["W1"] + self.params["b1"]
        self._a1 = self._activate(self._z1)
        if training and self.dropout_rate > 0:
            self._drop_mask = (np.random.rand(*self._a1.shape) >= self.dropout_rate) / (1 - self.dropout_rate)
            self._a1 = self._a1 * self._drop_mask
        out = self._a1 @ self.params["W2"] + self.params["b2"]
        return out

    def backward(self, grads, learning_rate):
        N = grads.shape[0]
        dparams = {}
        da1 = grads @ self.params["W2"].T
        dparams["dW2"] = self._a1.reshape(-1, self.d_ff).T @ grads.reshape(-1, self.d_model) / N
        dparams["db2"] = np.sum(grads, axis=tuple(range(grads.ndim - 1)), keepdims=True) / N
        if hasattr(self, '_drop_mask') and self._drop_mask is not None:
            da1 = da1 * self._drop_mask
        dz1 = self._activate_backward(self._z1, da1)
        dparams["dW1"] = self.inputs.reshape(-1, self.d_model).T @ dz1.reshape(-1, self.d_ff) / N
        dparams["db1"] = np.sum(dz1, axis=tuple(range(dz1.ndim - 1)), keepdims=True) / N
        dinputs = dz1 @ self.params["W1"].T
        if self.trainable:
            self.optimizer.update(self, dparams, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return sum(np.prod(v.shape) for v in self.params.values())

    def summary(self):
        return ["FeedForward", self.output_shape[1:], self.get_num_parameters(),
                self.activation]


class TransformerEncoderBlock(BaseLayer):
    def __init__(self, d_model, num_heads, d_ff=None, dropout_rate=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or d_model * 4
        self.dropout_rate = dropout_rate
        self.layer_type = "TransformerEncoderBlock"
        self.params = {}
        from .attention import MultiHeadAttention
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm()
        self.ff = FeedForward(d_model, self.d_ff, dropout_rate=dropout_rate)
        self.norm2 = LayerNorm()
        self.drop1 = Dropout(dropout_rate) if dropout_rate > 0 else None
        self.drop2 = Dropout(dropout_rate) if dropout_rate > 0 else None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.attn.build(input_shape)
        self.norm1.build(input_shape)
        self.ff.build(input_shape)
        self.norm2.build(input_shape)
        if self.drop1:
            self.drop1.build(input_shape)
            self.drop2.build(input_shape)

    def _init_sub_optimizers(self):
        if self.optimizer:
            for sub in [self.attn, self.norm1, self.ff, self.norm2]:
                if not sub.optimizer:
                    sub.optimizer = self.optimizer
                    self.optimizer.init_params(sub)

    def forward(self, inputs, training=True):
        self._init_sub_optimizers()
        self._residual1 = inputs
        attn_out = self.attn.forward(inputs, training)
        if self.drop1 and training:
            attn_out = self.drop1.forward(attn_out, training)
        self._normed1 = self.norm1.forward(inputs + attn_out, training)
        self._residual2 = self._normed1
        ff_out = self.ff.forward(self._normed1, training)
        if self.drop2 and training:
            ff_out = self.drop2.forward(ff_out, training)
        output = self.norm2.forward(self._normed1 + ff_out, training)
        return output

    def backward(self, grads, learning_rate):
        dnorm2 = self.norm2.backward(grads, learning_rate)
        dff = self.ff.backward(dnorm2, learning_rate)
        if self.drop2 and self.drop2.mask is not None:
            dff = self.drop2.backward(dff, learning_rate)
        dnorm1_input = dnorm2 + dff
        dnorm1 = self.norm1.backward(dnorm1_input, learning_rate)
        dattn = self.attn.backward(dnorm1, learning_rate)
        if self.drop1 and self.drop1.mask is not None:
            dattn = self.drop1.backward(dattn, learning_rate)
        return dnorm1 + dattn

    def get_num_parameters(self):
        return (self.attn.get_num_parameters() + self.norm1.get_num_parameters() +
                self.ff.get_num_parameters() + self.norm2.get_num_parameters())

    def summary(self):
        return ["TransEncBlock", self.output_shape[1:], self.get_num_parameters(),
                f"{self.num_heads}h d_ff={self.d_ff}"]


class TransformerDecoderBlock(BaseLayer):
    def __init__(self, d_model, num_heads, d_ff=None, dropout_rate=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or d_model * 4
        self.dropout_rate = dropout_rate
        self.layer_type = "TransformerDecoderBlock"
        self.params = {}
        from .attention import CausalMultiHeadAttention, CrossAttention
        self.self_attn = CausalMultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm()
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.norm2 = LayerNorm()
        self.ff = FeedForward(d_model, self.d_ff, dropout_rate=dropout_rate)
        self.norm3 = LayerNorm()
        self._encoder_output = None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.self_attn.build(input_shape)
        self.norm1.build(input_shape)
        self.cross_attn.build(input_shape)
        self.norm2.build(input_shape)
        self.ff.build(input_shape)
        self.norm3.build(input_shape)

    def set_encoder_output(self, encoder_output):
        self._encoder_output = encoder_output

    def _init_sub_optimizers(self):
        if self.optimizer:
            for sub in [self.self_attn, self.norm1, self.cross_attn, self.norm2, self.ff, self.norm3]:
                if not sub.optimizer:
                    sub.optimizer = self.optimizer
                    self.optimizer.init_params(sub)

    def forward(self, inputs, training=True):
        self._init_sub_optimizers()
        self._residual1 = inputs
        sa_out = self.self_attn.forward(inputs, training)
        x = self.norm1.forward(inputs + sa_out, training)
        self._residual2 = x
        if self._encoder_output is not None:
            ca_out = self.cross_attn.forward(x, self._encoder_output, training)
            x = self.norm2.forward(x + ca_out, training)
        self._residual3 = x
        ff_out = self.ff.forward(x, training)
        output = self.norm3.forward(x + ff_out, training)
        return output

    def backward(self, grads, learning_rate):
        dnorm3 = self.norm3.backward(grads, learning_rate)
        dff = self.ff.backward(dnorm3, learning_rate)
        dx = dnorm3 + dff
        if self._encoder_output is not None:
            dnorm2 = self.norm2.backward(dx, learning_rate)
            dca = self.cross_attn.backward(dnorm2, learning_rate)
            dx = dnorm2 + dca
        dnorm1 = self.norm1.backward(dx, learning_rate)
        dsa = self.self_attn.backward(dnorm1, learning_rate)
        return dnorm1 + dsa

    def get_num_parameters(self):
        total = (self.self_attn.get_num_parameters() + self.norm1.get_num_parameters() +
                 self.cross_attn.get_num_parameters() + self.norm2.get_num_parameters() +
                 self.ff.get_num_parameters() + self.norm3.get_num_parameters())
        return total

    def summary(self):
        return ["TransDecBlock", self.output_shape[1:], self.get_num_parameters(),
                f"{self.num_heads}h d_ff={self.d_ff}"]
