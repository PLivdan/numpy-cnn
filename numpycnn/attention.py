import numpy as np
from .layers import BaseLayer


class PositionalEncoding(BaseLayer):
    def __init__(self, max_len=5000):
        super().__init__()
        self.max_len = max_len
        self.layer_type = "PositionalEncoding"

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape
        d_model = input_shape[-1]
        pe = np.zeros((self.max_len, d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        self.pe = pe

    def forward(self, inputs, training=True):
        seq_len = inputs.shape[1]
        return inputs + self.pe[:seq_len]

    def backward(self, grads, learning_rate):
        return grads

    def get_num_parameters(self):
        return 0

    def summary(self):
        return ["PosEncoding", self.output_shape[1:], 0, ""]


class MultiHeadAttention(BaseLayer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.layer_type = "MultiHeadAttention"
        self.params = {"Wq": None, "Wk": None, "Wv": None, "Wo": None,
                       "bq": None, "bk": None, "bv": None, "bo": None}

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = input_shape
        d = self.d_model
        scale = np.sqrt(2.0 / d)
        self.params["Wq"] = np.random.randn(d, d) * scale
        self.params["Wk"] = np.random.randn(d, d) * scale
        self.params["Wv"] = np.random.randn(d, d) * scale
        self.params["Wo"] = np.random.randn(d, d) * scale
        self.params["bq"] = np.zeros((1, 1, d))
        self.params["bk"] = np.zeros((1, 1, d))
        self.params["bv"] = np.zeros((1, 1, d))
        self.params["bo"] = np.zeros((1, 1, d))

    def _split_heads(self, x):
        N, L, _ = x.shape
        return x.reshape(N, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        N, _, L, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, L, self.d_model)

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, L, D = inputs.shape

        self.Q = inputs @ self.params["Wq"] + self.params["bq"]
        self.K = inputs @ self.params["Wk"] + self.params["bk"]
        self.V = inputs @ self.params["Wv"] + self.params["bv"]

        self.Q_h = self._split_heads(self.Q)
        self.K_h = self._split_heads(self.K)
        self.V_h = self._split_heads(self.V)

        scores = self.Q_h @ self.K_h.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        self.attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attn_weights = self.attn_weights / np.sum(self.attn_weights, axis=-1, keepdims=True)

        self.attn_out = self.attn_weights @ self.V_h
        concat = self._merge_heads(self.attn_out)
        output = concat @ self.params["Wo"] + self.params["bo"]
        return output

    def backward(self, grads, learning_rate):
        N, L, D = self.inputs.shape
        dparams = {}

        dconcat = grads @ self.params["Wo"].T
        dparams["dWo"] = self._merge_heads(self.attn_out).reshape(-1, D).T @ grads.reshape(-1, D) / N
        dparams["dbo"] = np.sum(grads, axis=(0, 1), keepdims=True)

        dattn_out = self._split_heads(dconcat)
        dV_h = self.attn_weights.transpose(0, 1, 3, 2) @ dattn_out
        dattn_w = dattn_out @ self.V_h.transpose(0, 1, 3, 2)

        dscores = self.attn_weights * (dattn_w - np.sum(dattn_w * self.attn_weights, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(self.d_k)

        dQ_h = dscores @ self.K_h
        dK_h = dscores.transpose(0, 1, 3, 2) @ self.Q_h

        dQ = self._merge_heads(dQ_h)
        dK = self._merge_heads(dK_h)
        dV = self._merge_heads(dV_h)

        dparams["dWq"] = self.inputs.reshape(-1, D).T @ dQ.reshape(-1, D) / N
        dparams["dWk"] = self.inputs.reshape(-1, D).T @ dK.reshape(-1, D) / N
        dparams["dWv"] = self.inputs.reshape(-1, D).T @ dV.reshape(-1, D) / N
        dparams["dbq"] = np.sum(dQ, axis=(0, 1), keepdims=True)
        dparams["dbk"] = np.sum(dK, axis=(0, 1), keepdims=True)
        dparams["dbv"] = np.sum(dV, axis=(0, 1), keepdims=True)

        dinputs = dQ @ self.params["Wq"].T + dK @ self.params["Wk"].T + dV @ self.params["Wv"].T
        self.optimizer.update(self, dparams, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return sum(np.prod(v.shape) for v in self.params.values())

    def summary(self):
        return ["MultiHeadAttn", self.output_shape[1:], self.get_num_parameters(),
                f"{self.num_heads}h"]
