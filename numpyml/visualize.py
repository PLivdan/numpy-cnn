import numpy as np


def plot_history(history, figsize=(12, 4)):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(history['train_accuracy'], 'o-', label='Train', markersize=3)
    ax1.plot(history['val_accuracy'], 's-', label='Val', markersize=3)
    ax1.set_title('Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_loss'], 'o-', label='Train', markersize=3)
    ax2.plot(history['val_loss'], 's-', label='Val', markersize=3)
    ax2.set_title('Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    import matplotlib.pyplot as plt
    from .metrics import confusion_matrix as cm_fn
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    cm = cm_fn(y_true, y_pred)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold')
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.show()


def plot_feature_maps(feature_maps, max_filters=32, cmap='viridis', figsize=(14, 6)):
    import matplotlib.pyplot as plt
    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]
    n_filters = min(feature_maps.shape[-1], max_filters)
    cols = min(8, n_filters)
    rows = int(np.ceil(n_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes[np.newaxis, :]
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_filters:
            ax.imshow(feature_maps[:, :, i], cmap=cmap)
        ax.axis('off')
    plt.suptitle(f'Feature Maps ({n_filters} filters)', fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_kernels(weights, cmap='RdBu_r', figsize=(12, 6)):
    import matplotlib.pyplot as plt
    if weights.ndim == 4:
        n_filters = weights.shape[3]
        cols = min(8, n_filters)
        rows = int(np.ceil(n_filters / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < n_filters:
                k = weights[:, :, 0, i] if weights.shape[2] == 1 else np.mean(weights[:, :, :, i], axis=2)
                vmax = max(abs(k.min()), abs(k.max()))
                ax.imshow(k, cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_title(f'#{i}', fontsize=7)
            ax.axis('off')
        plt.suptitle(f'Learned Kernels ({weights.shape[0]}x{weights.shape[1]}, {n_filters} filters)', fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_attention_weights(attn_weights, head=None, figsize=(12, 4)):
    import matplotlib.pyplot as plt
    if attn_weights.ndim == 4:
        attn_weights = attn_weights[0]
    n_heads = attn_weights.shape[0]
    if head is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_weights[head], cmap='hot')
        ax.set_title(f'Head {head}', fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        fig.colorbar(im)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(1, n_heads, figsize=figsize)
        if n_heads == 1:
            axes = [axes]
        for h, ax in enumerate(axes):
            im = ax.imshow(attn_weights[h], cmap='hot')
            ax.set_title(f'Head {h}', fontweight='bold')
            ax.set_xlabel('Key')
            if h == 0:
                ax.set_ylabel('Query')
        fig.colorbar(im, ax=axes, fraction=0.02)
        plt.suptitle('Attention Weights', fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_predictions(images, predictions, true_labels=None, class_names=None, n=8, figsize=(16, 4)):
    import matplotlib.pyplot as plt
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=figsize)
    for i in range(n):
        ax = axes[i]
        img = np.squeeze(images[i])
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        pred = np.argmax(predictions[i])
        conf = predictions[i, pred]
        name = class_names[pred] if class_names else str(pred)
        color = 'green'
        if true_labels is not None:
            true = true_labels[i] if np.isscalar(true_labels[i]) else np.argmax(true_labels[i])
            if pred != true:
                color = 'red'
        ax.set_title(f'{name}\n{conf:.0%}', fontsize=9, color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
