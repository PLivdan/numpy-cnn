import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=1000, l2_lambda=0.01):
        self.lr = lr
        self.max_iter = max_iter
        self.l2_lambda = l2_lambda
        self.W = None
        self.b = None

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = len(np.unique(y))
        self.W = np.random.randn(d, self.n_classes) * 0.01
        self.b = np.zeros((1, self.n_classes))
        y_oh = np.eye(self.n_classes)[y]
        for _ in range(self.max_iter):
            proba = self._softmax(X @ self.W + self.b)
            dW = X.T @ (proba - y_oh) / n + self.l2_lambda * self.W
            db = np.sum(proba - y_oh, axis=0, keepdims=True) / n
            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return np.argmax(self._softmax(X @ self.W + self.b), axis=1)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class LinearRegression:
    def __init__(self, l2_lambda=0.0):
        self.l2_lambda = l2_lambda
        self.W = None

    def fit(self, X, y):
        X_b = np.c_[np.ones(len(X)), X]
        if self.l2_lambda > 0:
            I = np.eye(X_b.shape[1])
            I[0, 0] = 0
            self.W = np.linalg.solve(X_b.T @ X_b + self.l2_lambda * I, X_b.T @ y)
        else:
            self.W = np.linalg.lstsq(X_b, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.W

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)
