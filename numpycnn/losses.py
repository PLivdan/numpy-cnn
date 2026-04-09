import numpy as np


def mse_loss(predictions, targets):
    m = targets.shape[0]
    loss = np.mean((predictions - targets) ** 2)
    grads = (2 / m) * (predictions - targets)
    return loss, grads


def categorical_crossentropy(predictions, targets):
    m = targets.shape[0]
    clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
    loss = -np.sum(targets * np.log(clipped)) / m
    grads = predictions - targets
    return loss, grads


def huber_loss(predictions, targets, delta=1.0):
    residual = predictions - targets
    abs_residual = np.abs(residual)
    mask = abs_residual <= delta
    loss = np.mean(np.where(mask, 0.5 * residual ** 2, delta * (abs_residual - 0.5 * delta)))
    grads = np.where(mask, residual, delta * np.sign(residual))
    return loss, grads


def binary_crossentropy(predictions, targets):
    m = targets.shape[0]
    clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
    loss = -np.sum(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)) / m
    grads = (predictions - targets) / (predictions * (1 - predictions) + 1e-7)
    return loss, grads


def focal_loss(predictions, targets, gamma=2):
    m = targets.shape[0]
    clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
    loss = -np.sum(targets * np.power(1 - clipped, gamma) * np.log(clipped)) / m
    grads = -(targets - clipped) * np.power(1 - clipped, gamma - 1)
    return loss, grads


def label_smoothing_crossentropy(predictions, targets, smoothing=0.1):
    n_classes = targets.shape[-1]
    smooth_targets = targets * (1 - smoothing) + smoothing / n_classes
    return categorical_crossentropy(predictions, smooth_targets)


def kl_divergence(predictions, targets):
    m = targets.shape[0]
    clipped_pred = np.clip(predictions, 1e-7, 1 - 1e-7)
    clipped_targ = np.clip(targets, 1e-7, 1 - 1e-7)
    loss = np.sum(clipped_targ * np.log(clipped_targ / clipped_pred)) / m
    grads = -clipped_targ / clipped_pred / m
    return loss, grads


def hinge_loss(predictions, targets):
    targets_signed = 2 * targets - 1
    margins = 1 - targets_signed * predictions
    loss = np.mean(np.maximum(0, margins))
    grads = np.where(margins > 0, -targets_signed, 0) / targets.shape[0]
    return loss, grads
