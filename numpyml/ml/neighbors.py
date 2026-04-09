import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.n_classes = len(np.unique(y))
        return self

    def _distances(self, X):
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        TT = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
        return np.sqrt(np.maximum(XX + TT - 2 * X @ self.X_train.T, 0))

    def predict(self, X):
        dists = self._distances(X)
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            neighbor_labels = self.y_train[idx[i]]
            if self.weights == 'distance':
                w = 1 / (dists[i, idx[i]] + 1e-8)
                votes = np.bincount(neighbor_labels, weights=w, minlength=self.n_classes)
            else:
                votes = np.bincount(neighbor_labels, minlength=self.n_classes)
            preds[i] = np.argmax(votes)
        return preds

    def predict_proba(self, X):
        dists = self._distances(X)
        idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        probas = np.zeros((len(X), self.n_classes))
        for i in range(len(X)):
            neighbor_labels = self.y_train[idx[i]]
            if self.weights == 'distance':
                w = 1 / (dists[i, idx[i]] + 1e-8)
                probas[i] = np.bincount(neighbor_labels, weights=w, minlength=self.n_classes)
            else:
                probas[i] = np.bincount(neighbor_labels, minlength=self.n_classes)
            probas[i] /= probas[i].sum()
        return probas

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
