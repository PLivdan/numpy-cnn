import numpy as np
from .layers import BaseLayer


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class RNN(BaseLayer):
    def __init__(self, units, return_sequences=False, initializer="xavier"):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.layer_type = "RNN"
        self.params = {"W": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, T, D = input_shape
        H = self.units
        scale = np.sqrt(2.0 / (D + H))
        self.params["W"] = np.random.randn(D + H, H) * scale
        self.params["b"] = np.zeros((1, H))
        self.output_shape = (N, T, H) if self.return_sequences else (N, H)

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, T, D = inputs.shape
        H = self.units
        self.h = np.zeros((N, T + 1, H))
        self._xh = np.empty((N, D + H))
        for t in range(T):
            self._xh[:, :D] = inputs[:, t]
            self._xh[:, D:] = self.h[:, t]
            self.h[:, t + 1] = np.tanh(self._xh @ self.params["W"] + self.params["b"])
        if self.return_sequences:
            return self.h[:, 1:]
        return self.h[:, -1]

    def backward(self, grads, learning_rate):
        N, T, D = self.inputs.shape
        H = self.units
        dW = np.zeros_like(self.params["W"])
        db = np.zeros_like(self.params["b"])
        dinputs = np.zeros_like(self.inputs)
        if self.return_sequences:
            dh_all = grads
        else:
            dh_all = np.zeros((N, T, H))
            dh_all[:, -1] = grads
        dh_next = np.zeros((N, H))
        xh = np.empty((N, D + H))
        for t in reversed(range(T)):
            dh = dh_all[:, t] + dh_next
            dtanh = dh * (1 - self.h[:, t + 1] ** 2)
            xh[:, :D] = self.inputs[:, t]
            xh[:, D:] = self.h[:, t]
            dW += xh.T @ dtanh / N
            db += np.sum(dtanh, axis=0, keepdims=True) / N
            dxh = dtanh @ self.params["W"].T
            dinputs[:, t] = dxh[:, :D]
            dh_next = dxh[:, D:]
        if self.trainable:
            self.optimizer.update(self, {"dW": dW, "db": db}, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return sum(np.prod(v.shape) for v in self.params.values())

    def summary(self):
        return ["RNN", self.output_shape[1:], self.get_num_parameters(),
                f"h={self.units}" + (" seq" if self.return_sequences else "")]


class LSTM(BaseLayer):
    def __init__(self, units, return_sequences=False, initializer="xavier"):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.layer_type = "LSTM"
        self.params = {"W": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, T, D = input_shape
        H = self.units
        scale = np.sqrt(2.0 / (D + H))
        self.params["W"] = np.random.randn(D + H, 4 * H) * scale
        self.params["b"] = np.zeros((1, 4 * H))
        self.params["b"][0, H:2*H] = 1.0
        self.output_shape = (N, T, H) if self.return_sequences else (N, H)

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, T, D = inputs.shape
        H = self.units
        self.h = np.zeros((N, T + 1, H))
        self.c = np.zeros((N, T + 1, H))
        self._f = np.empty((T, N, H))
        self._i = np.empty((T, N, H))
        self._g = np.empty((T, N, H))
        self._o = np.empty((T, N, H))
        self._tanh_c = np.empty((T, N, H))
        xh = np.empty((N, D + H))
        W, b = self.params["W"], self.params["b"]
        for t in range(T):
            xh[:, :D] = inputs[:, t]
            xh[:, D:] = self.h[:, t]
            z = xh @ W + b
            self._f[t] = _sigmoid(z[:, :H])
            self._i[t] = _sigmoid(z[:, H:2*H])
            self._g[t] = np.tanh(z[:, 2*H:3*H])
            self._o[t] = _sigmoid(z[:, 3*H:])
            self.c[:, t + 1] = self._f[t] * self.c[:, t] + self._i[t] * self._g[t]
            self._tanh_c[t] = np.tanh(self.c[:, t + 1])
            self.h[:, t + 1] = self._o[t] * self._tanh_c[t]
        if self.return_sequences:
            return self.h[:, 1:]
        return self.h[:, -1]

    def backward(self, grads, learning_rate):
        N, T, D = self.inputs.shape
        H = self.units
        dW = np.zeros_like(self.params["W"])
        db = np.zeros_like(self.params["b"])
        dinputs = np.zeros_like(self.inputs)
        if self.return_sequences:
            dh_all = grads
        else:
            dh_all = np.zeros((N, T, H))
            dh_all[:, -1] = grads
        dh_next = np.zeros((N, H))
        dc_next = np.zeros((N, H))
        xh = np.empty((N, D + H))
        dz = np.empty((N, 4 * H))
        W = self.params["W"]
        for t in reversed(range(T)):
            f, i, g, o = self._f[t], self._i[t], self._g[t], self._o[t]
            tanh_c = self._tanh_c[t]
            dh = dh_all[:, t] + dh_next
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            df = dc * self.c[:, t]
            di = dc * g
            dg = dc * i
            dc_next = dc * f
            dz[:, :H] = df * f * (1 - f)
            dz[:, H:2*H] = di * i * (1 - i)
            dz[:, 2*H:3*H] = dg * (1 - g ** 2)
            dz[:, 3*H:] = do * o * (1 - o)
            xh[:, :D] = self.inputs[:, t]
            xh[:, D:] = self.h[:, t]
            dW += xh.T @ dz / N
            db += np.sum(dz, axis=0, keepdims=True) / N
            dxh = dz @ W.T
            dinputs[:, t] = dxh[:, :D]
            dh_next = dxh[:, D:]
        if self.trainable:
            self.optimizer.update(self, {"dW": dW, "db": db}, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return sum(np.prod(v.shape) for v in self.params.values())

    def summary(self):
        return ["LSTM", self.output_shape[1:], self.get_num_parameters(),
                f"h={self.units}" + (" seq" if self.return_sequences else "")]


class GRU(BaseLayer):
    def __init__(self, units, return_sequences=False, initializer="xavier"):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.layer_type = "GRU"
        self.params = {"Wz": None, "Wn": None, "bz": None, "bn": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, T, D = input_shape
        H = self.units
        scale = np.sqrt(2.0 / (D + H))
        self.params["Wz"] = np.random.randn(D + H, 2 * H) * scale
        self.params["Wn"] = np.random.randn(D + H, H) * scale
        self.params["bz"] = np.zeros((1, 2 * H))
        self.params["bn"] = np.zeros((1, H))
        self.output_shape = (N, T, H) if self.return_sequences else (N, H)

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, T, D = inputs.shape
        H = self.units
        self.h = np.zeros((N, T + 1, H))
        self._z = np.empty((T, N, H))
        self._r = np.empty((T, N, H))
        self._n = np.empty((T, N, H))
        xh = np.empty((N, D + H))
        xrh = np.empty((N, D + H))
        Wz, Wn = self.params["Wz"], self.params["Wn"]
        bz, bn = self.params["bz"], self.params["bn"]
        for t in range(T):
            xh[:, :D] = inputs[:, t]
            xh[:, D:] = self.h[:, t]
            zr = _sigmoid(xh @ Wz + bz)
            self._z[t] = zr[:, :H]
            self._r[t] = zr[:, H:]
            xrh[:, :D] = inputs[:, t]
            xrh[:, D:] = self._r[t] * self.h[:, t]
            self._n[t] = np.tanh(xrh @ Wn + bn)
            self.h[:, t + 1] = (1 - self._z[t]) * self.h[:, t] + self._z[t] * self._n[t]
        if self.return_sequences:
            return self.h[:, 1:]
        return self.h[:, -1]

    def backward(self, grads, learning_rate):
        N, T, D = self.inputs.shape
        H = self.units
        dWz = np.zeros_like(self.params["Wz"])
        dWn = np.zeros_like(self.params["Wn"])
        dbz = np.zeros_like(self.params["bz"])
        dbn = np.zeros_like(self.params["bn"])
        dinputs = np.zeros_like(self.inputs)
        if self.return_sequences:
            dh_all = grads
        else:
            dh_all = np.zeros((N, T, H))
            dh_all[:, -1] = grads
        dh_next = np.zeros((N, H))
        xh = np.empty((N, D + H))
        xrh = np.empty((N, D + H))
        Wz, Wn = self.params["Wz"], self.params["Wn"]
        for t in reversed(range(T)):
            z, r, n = self._z[t], self._r[t], self._n[t]
            dh = dh_all[:, t] + dh_next
            dz = dh * (n - self.h[:, t]) * z * (1 - z)
            dn = dh * z * (1 - n ** 2)
            xrh[:, :D] = self.inputs[:, t]
            xrh[:, D:] = r * self.h[:, t]
            dWn += xrh.T @ dn / N
            dbn += np.sum(dn, axis=0, keepdims=True) / N
            dxrh = dn @ Wn.T
            dr = dxrh[:, D:] * self.h[:, t] * r * (1 - r)
            dzr = np.concatenate([dz, dr], axis=1)
            xh[:, :D] = self.inputs[:, t]
            xh[:, D:] = self.h[:, t]
            dWz += xh.T @ dzr / N
            dbz += np.sum(dzr, axis=0, keepdims=True) / N
            dxh_zr = dzr @ Wz.T
            dinputs[:, t] = dxh_zr[:, :D] + dxrh[:, :D]
            dh_next = dh * (1 - z) + dxh_zr[:, D:] + dxrh[:, D:] * r
        if self.trainable:
            self.optimizer.update(self, {"dWz": dWz, "dWn": dWn, "dbz": dbz, "dbn": dbn}, learning_rate)
        return dinputs

    def get_num_parameters(self):
        return sum(np.prod(v.shape) for v in self.params.values())

    def summary(self):
        return ["GRU", self.output_shape[1:], self.get_num_parameters(),
                f"h={self.units}" + (" seq" if self.return_sequences else "")]


class Bidirectional(BaseLayer):
    def __init__(self, layer, merge_mode='concat'):
        super().__init__()
        self.forward_layer = layer
        import copy
        self.backward_layer = copy.deepcopy(layer)
        self.merge_mode = merge_mode
        self.layer_type = "Bidirectional"
        self.params = {}

    def build(self, input_shape):
        self.input_shape = input_shape
        self.forward_layer.build(input_shape)
        self.backward_layer.build(input_shape)
        if self.optimizer:
            self.forward_layer.optimizer = self.optimizer
            self.backward_layer.optimizer = self.optimizer
            self.optimizer.init_params(self.forward_layer)
            self.optimizer.init_params(self.backward_layer)
        fwd_shape = self.forward_layer.output_shape
        if self.merge_mode == 'concat':
            if len(fwd_shape) == 3:
                self.output_shape = (fwd_shape[0], fwd_shape[1], fwd_shape[2] * 2)
            else:
                self.output_shape = (fwd_shape[0], fwd_shape[1] * 2)
        else:
            self.output_shape = fwd_shape

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if self.optimizer and not self.forward_layer.optimizer:
            self.forward_layer.optimizer = self.optimizer
            self.backward_layer.optimizer = self.optimizer
            self.optimizer.init_params(self.forward_layer)
            self.optimizer.init_params(self.backward_layer)
        fwd_out = self.forward_layer.forward(inputs, training)
        bwd_out = self.backward_layer.forward(inputs[:, ::-1], training)
        if self.forward_layer.return_sequences:
            bwd_out = bwd_out[:, ::-1]
        if self.merge_mode == 'concat':
            return np.concatenate([fwd_out, bwd_out], axis=-1)
        elif self.merge_mode == 'sum':
            return fwd_out + bwd_out
        elif self.merge_mode == 'mul':
            self._fwd_out = fwd_out
            self._bwd_out = bwd_out
            return fwd_out * bwd_out
        elif self.merge_mode == 'avg':
            return (fwd_out + bwd_out) / 2

    def backward(self, grads, learning_rate):
        if self.merge_mode == 'concat':
            H = grads.shape[-1] // 2
            fwd_grads = grads[..., :H]
            bwd_grads = grads[..., H:]
        elif self.merge_mode == 'sum':
            fwd_grads = grads
            bwd_grads = grads
        elif self.merge_mode == 'mul':
            fwd_grads = grads * self._bwd_out
            bwd_grads = grads * self._fwd_out
        elif self.merge_mode == 'avg':
            fwd_grads = grads / 2
            bwd_grads = grads / 2
        if self.forward_layer.return_sequences:
            bwd_grads = bwd_grads[:, ::-1]
        dinputs_fwd = self.forward_layer.backward(fwd_grads, learning_rate)
        dinputs_bwd = self.backward_layer.backward(bwd_grads, learning_rate)
        return dinputs_fwd + dinputs_bwd[:, ::-1]

    def get_num_parameters(self):
        return self.forward_layer.get_num_parameters() + self.backward_layer.get_num_parameters()

    def summary(self):
        return ["Bidirectional", self.output_shape[1:], self.get_num_parameters(),
                f"{self.forward_layer.layer_type}"]
