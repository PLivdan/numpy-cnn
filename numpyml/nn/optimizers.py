import numpy as np


class Optimizer:
    def __init__(self, use_agc=False, clip_factor=0.1, clip_norm=None, clip_value=None):
        self.use_agc = use_agc
        self.clip_factor = clip_factor
        self.clip_norm = clip_norm
        self.clip_value = clip_value

    def adaptive_gradient_clipping(self, layer, grads):
        for key in layer.params.keys():
            grad_norm = np.linalg.norm(grads["d" + key])
            param_norm = np.linalg.norm(layer.params[key])
            max_norm = self.clip_factor * (param_norm + 1e-8)
            clipping_factor = min(1, max_norm / (grad_norm + 1e-8))
            grads["d" + key] = clipping_factor * grads["d" + key]

    def gradient_clipping(self, grads):
        if self.clip_value is not None:
            for key in grads:
                grads[key] = np.clip(grads[key], -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            total_norm = np.sqrt(sum(np.sum(grads[k] ** 2) for k in grads))
            if total_norm > self.clip_norm:
                scale = self.clip_norm / (total_norm + 1e-8)
                for key in grads:
                    grads[key] = grads[key] * scale

    def init_params(self, layer):
        pass

    def update(self, layer, grads, lr):
        if self.use_agc:
            self.adaptive_gradient_clipping(layer, grads)
        if self.clip_norm is not None or self.clip_value is not None:
            self.gradient_clipping(grads)


class SGD(Optimizer):
    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        for key in layer.params.keys():
            layer.params[key] -= lr * grads["d" + key]


class SGDmom(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.momentum = 0.9

    def init_params(self, layer):
        layer.v = {}
        for key in layer.params.keys():
            layer.v["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        for key in layer.params.keys():
            layer.v["d" + key] = self.momentum * layer.v["d" + key] + (1 - self.momentum) * grads["d" + key]
            layer.params[key] -= lr * layer.v["d" + key]


class RMSprop(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decay = 0.9
        self.epsilon = 1e-8

    def init_params(self, layer):
        layer.s = {}
        for key in layer.params.keys():
            layer.s["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        for key in layer.params.keys():
            layer.s["d" + key] = self.decay * layer.s["d" + key] + (1 - self.decay) * grads["d" + key]**2
            layer.params[key] -= lr * grads["d" + key] / (np.sqrt(layer.s["d" + key]) + self.epsilon)


class AdaGrad(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-8

    def init_params(self, layer):
        layer.h = {}
        for key in layer.params.keys():
            layer.h["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        for key in layer.params.keys():
            layer.h["d" + key] += grads["d" + key] ** 2
            layer.params[key] -= lr * grads["d" + key] / (np.sqrt(layer.h["d" + key]) + self.epsilon)


class AdaDelta(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decay = 0.9
        self.epsilon = 1e-8

    def init_params(self, layer):
        layer.h = {}
        layer.delta = {}
        for key in layer.params.keys():
            layer.h["d" + key] = np.zeros_like(layer.params[key])
            layer.delta["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        for key in layer.params.keys():
            layer.h["d" + key] = self.decay * layer.h["d" + key] + (1 - self.decay) * grads["d" + key]**2
            dx = -(np.sqrt(layer.delta["d" + key] + self.epsilon) / np.sqrt(layer.h["d" + key] + self.epsilon)) * grads["d" + key]
            layer.params[key] += dx
            layer.delta["d" + key] = self.decay * layer.delta["d" + key] + (1 - self.decay) * dx**2


class Adam(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def init_params(self, layer):
        layer.v = {}
        layer.s = {}
        layer.t = 0
        for key in layer.params.keys():
            layer.v["d" + key] = np.zeros_like(layer.params[key])
            layer.s["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        layer.t += 1
        v_corr = {}
        s_corr = {}
        for key in layer.params.keys():
            layer.v["d" + key] = self.beta1 * layer.v["d" + key] + (1 - self.beta1) * grads["d" + key]
            v_corr["d" + key] = layer.v["d" + key] / (1 - self.beta1 ** layer.t)
            layer.s["d" + key] = self.beta2 * layer.s["d" + key] + (1 - self.beta2) * np.square(grads["d" + key])
            s_corr["d" + key] = layer.s["d" + key] / (1 - self.beta2 ** layer.t)
            layer.params[key] -= lr * v_corr["d" + key] / (np.sqrt(s_corr["d" + key]) + self.epsilon)


class AdamW(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.weight_decay = 0.01

    def init_params(self, layer):
        layer.v = {}
        layer.s = {}
        layer.t = 0
        for key in layer.params.keys():
            layer.v["d" + key] = np.zeros_like(layer.params[key])
            layer.s["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        layer.t += 1
        for key in layer.params.keys():
            layer.v["d" + key] = self.beta1 * layer.v["d" + key] + (1 - self.beta1) * grads["d" + key]
            v_corr = layer.v["d" + key] / (1 - self.beta1 ** layer.t)
            layer.s["d" + key] = self.beta2 * layer.s["d" + key] + (1 - self.beta2) * np.square(grads["d" + key])
            s_corr = layer.s["d" + key] / (1 - self.beta2 ** layer.t)
            layer.params[key] -= lr * v_corr / (np.sqrt(s_corr) + self.epsilon)
            if key == 'W':
                layer.params[key] -= lr * self.weight_decay * layer.params[key]


class NAG(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.momentum = 0.9

    def init_params(self, layer):
        layer.v = {}
        for key in layer.params.keys():
            layer.v["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        for key in layer.params.keys():
            old_v = layer.v["d" + key]
            layer.v["d" + key] = self.momentum * layer.v["d" + key] - lr * grads["d" + key]
            layer.params[key] += -self.momentum * old_v + (1 + self.momentum) * layer.v["d" + key]


class AMSGrad(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def init_params(self, layer):
        layer.v = {}
        layer.s = {}
        layer.hat_s = {}
        layer.t = 0
        for key in layer.params.keys():
            layer.v["d" + key] = np.zeros_like(layer.params[key])
            layer.s["d" + key] = np.zeros_like(layer.params[key])
            layer.hat_s["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        layer.t += 1
        v_corr = {}
        for key in layer.params.keys():
            layer.v["d" + key] = self.beta1 * layer.v["d" + key] + (1 - self.beta1) * grads["d" + key]
            v_corr["d" + key] = layer.v["d" + key] / (1 - self.beta1 ** layer.t)
            layer.s["d" + key] = self.beta2 * layer.s["d" + key] + (1 - self.beta2) * np.square(grads["d" + key])
            layer.hat_s["d" + key] = np.maximum(layer.hat_s["d" + key], layer.s["d" + key])
            layer.params[key] -= lr * v_corr["d" + key] / (np.sqrt(layer.hat_s["d" + key]) + self.epsilon)


class Yogi(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-3

    def init_params(self, layer):
        layer.v = {}
        layer.s = {}
        layer.t = 0
        for key in layer.params.keys():
            layer.v["d" + key] = np.zeros_like(layer.params[key])
            layer.s["d" + key] = np.zeros_like(layer.params[key])

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        layer.t += 1
        v_corr = {}
        for key in layer.params.keys():
            layer.v["d" + key] = self.beta1 * layer.v["d" + key] + (1 - self.beta1) * grads["d" + key]
            v_corr["d" + key] = layer.v["d" + key] / (1 - self.beta1 ** layer.t)
            sign = np.sign(layer.s["d" + key] - grads["d" + key] ** 2)
            layer.s["d" + key] = self.beta2 * layer.s["d" + key] + (1 - self.beta2) * sign * grads["d" + key] ** 2
            layer.s["d" + key] = np.abs(layer.s["d" + key])
            layer.params[key] -= lr * v_corr["d" + key] / (np.sqrt(layer.s["d" + key]) + self.epsilon)


class AdaFactor(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon1 = 1e-30
        self.epsilon2 = 1e-3
        self.decay = 0.999

    def init_params(self, layer):
        layer.v_row = {}
        layer.v_col = {}
        layer.t = 0
        for key in layer.params.keys():
            if len(layer.params[key].shape) == 2:
                layer.v_row["d" + key] = np.zeros(layer.params[key].shape[0])
                layer.v_col["d" + key] = np.zeros(layer.params[key].shape[1])
            else:
                layer.v_row["d" + key] = np.zeros_like(layer.params[key])
                layer.v_col["d" + key] = None

    def update(self, layer, grads, lr):
        super().update(layer, grads, lr)
        layer.t += 1
        for key in layer.params.keys():
            grad = grads["d" + key]
            if len(grad.shape) == 2:
                layer.v_row["d" + key] = self.decay * layer.v_row["d" + key] + (1 - self.decay) * np.sum(grad ** 2, axis=1)
                layer.v_col["d" + key] = self.decay * layer.v_col["d" + key] + (1 - self.decay) * np.sum(grad ** 2, axis=0)
                v_row = np.sqrt(layer.v_row["d" + key] + self.epsilon1).reshape(-1, 1)
                v_col = np.sqrt(layer.v_col["d" + key] + self.epsilon1).reshape(1, -1)
                factor_v = grad / (v_row * v_col + self.epsilon2)
            else:
                v_row = np.sqrt(layer.v_row["d" + key] + self.epsilon1)
                factor_v = grad / (v_row + self.epsilon2)
            layer.params[key] -= lr * factor_v
