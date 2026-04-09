import numpy as np
from .layers import BaseLayer


class RNN(BaseLayer):
    def __init__(self, units, return_sequences=False, initializer="xavier"):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.layer_type = "RNN"
        self.params = {"W_ih": None, "W_hh": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, T, D = input_shape
        H = self.units
        scale = np.sqrt(2.0 / (D + H))
        self.params["W_ih"] = np.random.randn(D, H) * scale
        self.params["W_hh"] = np.random.randn(H, H) * scale
        self.params["b"] = np.zeros((1, H))
        self.output_shape = (N, T, H) if self.return_sequences else (N, H)

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, T, D = inputs.shape
        H = self.units
        self.h = np.zeros((N, T + 1, H))
        for t in range(T):
            self.h[:, t + 1] = np.tanh(
                inputs[:, t] @ self.params["W_ih"] +
                self.h[:, t] @ self.params["W_hh"] +
                self.params["b"]
            )
        if self.return_sequences:
            return self.h[:, 1:]
        return self.h[:, -1]

    def backward(self, grads, learning_rate):
        N, T, D = self.inputs.shape
        H = self.units
        dparams = {
            "dW_ih": np.zeros_like(self.params["W_ih"]),
            "dW_hh": np.zeros_like(self.params["W_hh"]),
            "db": np.zeros_like(self.params["b"]),
        }
        dinputs = np.zeros_like(self.inputs)
        if self.return_sequences:
            dh_all = grads
        else:
            dh_all = np.zeros((N, T, H))
            dh_all[:, -1] = grads
        dh_next = np.zeros((N, H))
        for t in reversed(range(T)):
            dh = dh_all[:, t] + dh_next
            dtanh = dh * (1 - self.h[:, t + 1] ** 2)
            dparams["dW_ih"] += self.inputs[:, t].T @ dtanh / N
            dparams["dW_hh"] += self.h[:, t].T @ dtanh / N
            dparams["db"] += np.sum(dtanh, axis=0, keepdims=True) / N
            dinputs[:, t] = dtanh @ self.params["W_ih"].T
            dh_next = dtanh @ self.params["W_hh"].T
        if self.trainable:
            self.optimizer.update(self, dparams, learning_rate)
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
        self.params = {"W_i": None, "W_h": None, "b": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, T, D = input_shape
        H = self.units
        scale = np.sqrt(2.0 / (D + H))
        self.params["W_i"] = np.random.randn(D, 4 * H) * scale
        self.params["W_h"] = np.random.randn(H, 4 * H) * scale
        self.params["b"] = np.zeros((1, 4 * H))
        self.params["b"][0, H:2*H] = 1.0
        self.output_shape = (N, T, H) if self.return_sequences else (N, H)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, T, D = inputs.shape
        H = self.units
        self.h = np.zeros((N, T + 1, H))
        self.c = np.zeros((N, T + 1, H))
        self.gates = {}
        for t in range(T):
            z = inputs[:, t] @ self.params["W_i"] + self.h[:, t] @ self.params["W_h"] + self.params["b"]
            f = self._sigmoid(z[:, :H])
            i = self._sigmoid(z[:, H:2*H])
            g = np.tanh(z[:, 2*H:3*H])
            o = self._sigmoid(z[:, 3*H:])
            self.c[:, t + 1] = f * self.c[:, t] + i * g
            self.h[:, t + 1] = o * np.tanh(self.c[:, t + 1])
            self.gates[t] = (f, i, g, o)
        if self.return_sequences:
            return self.h[:, 1:]
        return self.h[:, -1]

    def backward(self, grads, learning_rate):
        N, T, D = self.inputs.shape
        H = self.units
        dparams = {
            "dW_i": np.zeros_like(self.params["W_i"]),
            "dW_h": np.zeros_like(self.params["W_h"]),
            "db": np.zeros_like(self.params["b"]),
        }
        dinputs = np.zeros_like(self.inputs)
        if self.return_sequences:
            dh_all = grads
        else:
            dh_all = np.zeros((N, T, H))
            dh_all[:, -1] = grads
        dh_next = np.zeros((N, H))
        dc_next = np.zeros((N, H))
        for t in reversed(range(T)):
            f, i, g, o = self.gates[t]
            dh = dh_all[:, t] + dh_next
            tanh_c = np.tanh(self.c[:, t + 1])
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            df = dc * self.c[:, t]
            di = dc * g
            dg = dc * i
            dc_next = dc * f
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            dg_raw = dg * (1 - g ** 2)
            do_raw = do * o * (1 - o)
            dz = np.concatenate([df_raw, di_raw, dg_raw, do_raw], axis=1)
            dparams["dW_i"] += self.inputs[:, t].T @ dz / N
            dparams["dW_h"] += self.h[:, t].T @ dz / N
            dparams["db"] += np.sum(dz, axis=0, keepdims=True) / N
            dinputs[:, t] = dz @ self.params["W_i"].T
            dh_next = dz @ self.params["W_h"].T
        if self.trainable:
            self.optimizer.update(self, dparams, learning_rate)
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
        self.params = {"W_i": None, "W_h": None, "b_i": None, "b_h": None}

    def build(self, input_shape):
        super().build(input_shape)
        N, T, D = input_shape
        H = self.units
        scale = np.sqrt(2.0 / (D + H))
        self.params["W_i"] = np.random.randn(D, 3 * H) * scale
        self.params["W_h"] = np.random.randn(H, 3 * H) * scale
        self.params["b_i"] = np.zeros((1, 3 * H))
        self.params["b_h"] = np.zeros((1, 3 * H))
        self.output_shape = (N, T, H) if self.return_sequences else (N, H)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs, training=True):
        self.inputs = inputs
        N, T, D = inputs.shape
        H = self.units
        self.h = np.zeros((N, T + 1, H))
        self.cache = {}
        for t in range(T):
            x_gates = inputs[:, t] @ self.params["W_i"] + self.params["b_i"]
            h_gates = self.h[:, t] @ self.params["W_h"] + self.params["b_h"]
            z = self._sigmoid(x_gates[:, :H] + h_gates[:, :H])
            r = self._sigmoid(x_gates[:, H:2*H] + h_gates[:, H:2*H])
            n = np.tanh(x_gates[:, 2*H:] + r * h_gates[:, 2*H:])
            self.h[:, t + 1] = (1 - z) * self.h[:, t] + z * n
            self.cache[t] = (z, r, n, x_gates, h_gates)
        if self.return_sequences:
            return self.h[:, 1:]
        return self.h[:, -1]

    def backward(self, grads, learning_rate):
        N, T, D = self.inputs.shape
        H = self.units
        dparams = {
            "dW_i": np.zeros_like(self.params["W_i"]),
            "dW_h": np.zeros_like(self.params["W_h"]),
            "db_i": np.zeros_like(self.params["b_i"]),
            "db_h": np.zeros_like(self.params["b_h"]),
        }
        dinputs = np.zeros_like(self.inputs)
        if self.return_sequences:
            dh_all = grads
        else:
            dh_all = np.zeros((N, T, H))
            dh_all[:, -1] = grads
        dh_next = np.zeros((N, H))
        for t in reversed(range(T)):
            z, r, n, x_gates, h_gates = self.cache[t]
            dh = dh_all[:, t] + dh_next
            dz = dh * (n - self.h[:, t])
            dn = dh * z
            dh_prev_z = dh * (1 - z)
            dn_raw = dn * (1 - n ** 2)
            dz_raw = dz * z * (1 - z)
            dx_n = dn_raw
            dh_n = dn_raw * r
            dr = dn_raw * h_gates[:, 2*H:]
            dr_raw = dr * r * (1 - r)
            dx_gates = np.zeros_like(x_gates)
            dx_gates[:, :H] = dz_raw
            dx_gates[:, H:2*H] = dr_raw
            dx_gates[:, 2*H:] = dx_n
            dh_gates = np.zeros_like(h_gates)
            dh_gates[:, :H] = dz_raw
            dh_gates[:, H:2*H] = dr_raw
            dh_gates[:, 2*H:] = dh_n
            dparams["dW_i"] += self.inputs[:, t].T @ dx_gates / N
            dparams["dW_h"] += self.h[:, t].T @ dh_gates / N
            dparams["db_i"] += np.sum(dx_gates, axis=0, keepdims=True) / N
            dparams["db_h"] += np.sum(dh_gates, axis=0, keepdims=True) / N
            dinputs[:, t] = dx_gates @ self.params["W_i"].T
            dh_next = dh_prev_z + dh_gates @ self.params["W_h"].T
        if self.trainable:
            self.optimizer.update(self, dparams, learning_rate)
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
