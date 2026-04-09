import numpy as np


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = X_centered.T @ X_centered / (len(X) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[idx][:self.n_components]
        self.components = eigenvectors[:, idx[:self.n_components]]
        self.explained_variance_ratio = self.explained_variance / eigenvalues.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components.T + self.mean
