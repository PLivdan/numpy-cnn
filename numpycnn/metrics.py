import numpy as np


def accuracy(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=None):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision(y_true, y_pred, average='macro', num_classes=None):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    p = tp / (tp + fp + 1e-8)
    if average == 'macro':
        return np.mean(p)
    return p


def recall(y_true, y_pred, average='macro', num_classes=None):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    r = tp / (tp + fn + 1e-8)
    if average == 'macro':
        return np.mean(r)
    return r


def f1_score(y_true, y_pred, average='macro', num_classes=None):
    p = precision(y_true, y_pred, average=average, num_classes=num_classes)
    r = recall(y_true, y_pred, average=average, num_classes=num_classes)
    return 2 * p * r / (p + r + 1e-8)


def top_k_accuracy(y_true, y_pred, k=5):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    top_k = np.argsort(y_pred, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])


def classification_report(y_true, y_pred, class_names=None):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    num_classes = max(y_true.max(), y_pred.max()) + 1
    p = precision(y_true, y_pred, average=None, num_classes=num_classes)
    r = recall(y_true, y_pred, average=None, num_classes=num_classes)
    f1 = 2 * p * r / (p + r + 1e-8)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    support = cm.sum(axis=1)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    for i in range(num_classes):
        name = class_names[i] if class_names else str(i)
        print(f"{name:<12} {p[i]:>10.4f} {r[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d}")
    print("-" * 54)
    print(f"{'Macro avg':<12} {np.mean(p):>10.4f} {np.mean(r):>10.4f} {np.mean(f1):>10.4f} {np.sum(support):>10d}")
    print(f"{'Accuracy':<12} {'':>10} {'':>10} {accuracy(y_true, y_pred):>10.4f} {np.sum(support):>10d}")
