import numpy as np


def gradient_check(model, X, y, loss_fn, epsilon=1e-5):
    predictions = model.forward(X, training=True)
    loss, grads = loss_fn(predictions, y)
    model.backward(grads, 0.0)

    results = []
    for i, layer in enumerate(model.layers):
        for key, param in layer.params.items():
            if param is None or not isinstance(param, np.ndarray):
                continue
            grad_key = "d" + key
            if not hasattr(layer, '_last_dparams'):
                continue
            analytical = layer._last_dparams.get(grad_key)
            if analytical is None:
                continue
            numerical = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            checked = 0
            while not it.finished and checked < 20:
                idx = it.multi_index
                old_val = param[idx]
                param[idx] = old_val + epsilon
                loss_plus, _ = loss_fn(model.forward(X, training=False), y)
                param[idx] = old_val - epsilon
                loss_minus, _ = loss_fn(model.forward(X, training=False), y)
                param[idx] = old_val
                numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                it.iternext()
                checked += 1
            if analytical is not None:
                diff = np.max(np.abs(analytical.ravel()[:20] - numerical.ravel()[:20]))
                norm = np.max(np.abs(analytical.ravel()[:20])) + np.max(np.abs(numerical.ravel()[:20])) + 1e-8
                rel_error = diff / norm
                results.append((i, layer.layer_type, key, rel_error))
    return results


def count_parameters(model):
    total = 0
    trainable = 0
    for layer in model.layers:
        n = layer.get_num_parameters()
        total += n
        if layer.trainable:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def weight_statistics(model):
    stats = []
    for i, layer in enumerate(model.layers):
        for key, param in layer.params.items():
            if param is None or not isinstance(param, np.ndarray):
                continue
            stats.append({
                "layer": i,
                "type": layer.layer_type,
                "param": key,
                "shape": param.shape,
                "mean": float(np.mean(param)),
                "std": float(np.std(param)),
                "min": float(np.min(param)),
                "max": float(np.max(param)),
                "norm": float(np.linalg.norm(param)),
                "zeros_pct": float(np.mean(param == 0) * 100),
            })
    return stats


def print_weight_statistics(model):
    stats = weight_statistics(model)
    print(f"{'Layer':<5} {'Type':<20} {'Param':<8} {'Shape':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("=" * 100)
    for s in stats:
        shape_str = str(s['shape'])
        print(f"{s['layer']:<5} {s['type']:<20} {s['param']:<8} {shape_str:<18} "
              f"{s['mean']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}")


def model_size_bytes(model):
    total = 0
    for layer in model.layers:
        for param in layer.params.values():
            if param is not None and isinstance(param, np.ndarray):
                total += param.nbytes
    return total


def model_size_human(model):
    b = model_size_bytes(model)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def set_seed(seed):
    np.random.seed(seed)


def compare_models(models, X_test, y_test, names=None):
    if names is None:
        names = [f"Model {i}" for i in range(len(models))]
    results = []
    for name, model in zip(names, models):
        preds = model.predict(X_test)
        pred_labels = np.argmax(preds, axis=1)
        true_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        acc = np.mean(pred_labels == true_labels)
        n_params = sum(l.get_num_parameters() for l in model.layers)
        results.append({"name": name, "accuracy": acc, "params": n_params,
                        "size": model_size_human(model)})
    print(f"{'Model':<20} {'Accuracy':>10} {'Params':>12} {'Size':>10}")
    print("=" * 55)
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['params']:>12,} {r['size']:>10}")
    return results
