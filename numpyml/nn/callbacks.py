class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, monitor='val_loss', restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best = restore_best
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped = False
        self.best_weights = None

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = self.best_value is None or current < self.best_value - self.min_delta
        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best and hasattr(self, '_model'):
                self.best_weights = self._snapshot(self._model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                print(f"Early stopping at epoch {epoch + 1}. Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch + 1}")
                if self.restore_best and self.best_weights and hasattr(self, '_model'):
                    self._restore(self._model, self.best_weights)
                    print("Restored best weights.")

    @staticmethod
    def _snapshot(model):
        import copy
        weights = []
        for layer in model.layers:
            weights.append({k: v.copy() if hasattr(v, 'copy') else v for k, v in layer.params.items()})
        return weights

    @staticmethod
    def _restore(model, weights):
        for layer, w in zip(model.layers, weights):
            for k, v in w.items():
                layer.params[k] = v


class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = None

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if not self.save_best_only or self.best_value is None or current < self.best_value:
            self.best_value = current
            if hasattr(self, '_model'):
                self._model.save(self.filepath)
