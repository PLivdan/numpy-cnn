import numpy as np
import pickle


class Model:
    def __init__(self):
        self.layers = []
        self.compiled = False
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, input_shape, optimizer, initializer='xavier'):
        if not optimizer:
            raise Exception("Optimizer must be set before compiling.")
        self.optimizer = optimizer
        self.initializer = initializer
        current_shape = input_shape
        for layer in self.layers:
            if hasattr(layer, 'initializer'):
                layer.initializer = self.initializer
            layer.build(current_shape)
            layer.optimizer = self.optimizer
            layer.optimizer.init_params(layer)
            current_shape = layer.output_shape
        self.compiled = True

    def forward(self, inputs, training=True, return_layer=None):
        layer_outputs = []
        current_output = inputs
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "SkipConnection":
                skip_input = layer_outputs[layer.skip_from]
                current_output = layer.forward(current_output, skip_input, training=training)
            else:
                current_output = layer.forward(current_output, training=training)
            layer_outputs.append(current_output)
            if return_layer is not None and i == return_layer:
                return current_output
        return current_output

    def backward(self, grads, learning_rate):
        current_grad = grads
        for layer in reversed(self.layers):
            if layer.layer_type == "SkipConnection":
                continue
            current_grad = layer.backward(current_grad, learning_rate)

    def summary(self):
        if not self.compiled:
            raise Exception("Model needs to be compiled before summary.")
        total_params = sum(layer.get_num_parameters() for layer in self.layers)
        print("Layer #   OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)   ACTIVATION")
        print("=" * 76)
        for i, layer in enumerate(self.layers):
            operation, data_dims, weights_n, activation = layer.summary()
            if operation is None:
                continue
            weights_perc = (weights_n / total_params * 100) if total_params > 0 else 0
            data_dims_str = 'x'.join(str(dim) for dim in data_dims)
            activation_str = activation.upper() if activation else ""
            print(f"{i:<8} {operation:<20} {data_dims_str:>15} {weights_n:>12} {weights_perc:>10.1f}% {activation_str}")
        print("=" * 76)
        print(f"Total Parameters: {total_params:,}")

    def train_on_batch(self, inputs, targets, loss_fn, learning_rate, l2_lambda=0.01):
        predictions = self.forward(inputs, training=True)
        loss, grads = loss_fn(predictions, targets)
        l2_loss = sum(layer.l2_regularization(l2_lambda) for layer in self.layers)
        loss += l2_loss / targets.shape[0]
        self.backward(grads, learning_rate)
        return loss

    def predict(self, inputs):
        return self.forward(inputs, training=False)

    def fit(self, X_train, y_train, X_val, y_val, batch_size, epochs, loss_fn,
            l2_lambda=0.01, lr_scheduler=None, checkpoint_path=None, callbacks=[], augmentor=None):
        if not self.compiled:
            raise Exception("Model needs to be compiled before training.")
        learning_rate = lr_scheduler.initial_lr if lr_scheduler else 0.01
        history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
        n_batches = int(np.ceil(X_train.shape[0] / batch_size))
        best_val_loss = float('inf')

        for cb in callbacks:
            if hasattr(cb, '_model'):
                cb._model = self

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            epoch_train_losses = []

            print(f"\n Epoch {epoch+1}/{epochs}")
            for i in range(n_batches):
                X_batch = X_train_shuffled[i*batch_size:(i+1)*batch_size]
                y_batch = y_train_shuffled[i*batch_size:(i+1)*batch_size]
                if augmentor:
                    X_batch = augmentor.augment(X_batch)
                batch_train_loss = self.train_on_batch(X_batch, y_batch, loss_fn, learning_rate, l2_lambda)
                epoch_train_losses.append(batch_train_loss)
                progress = (i+1) / n_batches
                filled_elements = int(progress * 40)
                bar = '=' * filled_elements + '>' + '.' * (39 - filled_elements)
                print(f"\r[{bar}] Batch {i+1}/{n_batches} - Loss: {batch_train_loss:.4f}", end="")

            train_predictions = np.concatenate(
                [self.predict(X_train[i*batch_size:(i+1)*batch_size])
                 for i in range(int(np.ceil(X_train.shape[0] / batch_size)))], axis=0)
            train_predicted_labels = np.argmax(train_predictions, axis=1)
            train_true_labels = np.argmax(y_train, axis=1)
            train_accuracy = np.mean(train_predicted_labels == train_true_labels)
            train_loss = np.mean(epoch_train_losses)

            val_predictions = np.concatenate(
                [self.predict(X_val[i*batch_size:(i+1)*batch_size])
                 for i in range(int(np.ceil(X_val.shape[0] / batch_size)))], axis=0)
            val_loss, _ = loss_fn(val_predictions, y_val)
            val_predicted_labels = np.argmax(val_predictions, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_predicted_labels == val_true_labels)

            print(f" - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")

            if checkpoint_path and val_loss < best_val_loss:
                self.save(checkpoint_path)
                best_val_loss = val_loss
                print(f"Checkpoint saved to {checkpoint_path}")

            if lr_scheduler:
                learning_rate = lr_scheduler(epoch, val_loss)

            logs = {
                'train_loss': train_loss, 'train_accuracy': train_accuracy,
                'val_loss': val_loss, 'val_accuracy': val_accuracy,
            }
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)

            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            if any(getattr(cb, 'stopped', False) for cb in callbacks):
                break
        return history

    @staticmethod
    def _get_init_args(layer):
        layer_type = type(layer).__name__
        arg_map = {
            'Conv2D': ('filters', 'kernel_size', 'stride', 'padding', 'activation', 'initializer'),
            'Conv1D': ('filters', 'kernel_size', 'stride', 'padding', 'activation', 'initializer'),
            'ConvTranspose2D': ('filters', 'kernel_size', 'stride', 'padding', 'activation', 'initializer'),
            'DepthwiseConv2D': ('kernel_size', 'stride', 'padding', 'depth_multiplier', 'initializer'),
            'SeparableConv2D': ('filters', 'kernel_size', 'stride', 'padding', 'activation', 'depth_multiplier', 'initializer'),
            'Pooling2D': ('pool_size', 'stride', 'mode'),
            'Pooling1D': ('pool_size', 'stride', 'mode'),
            'GlobalAvgPool2D': (),
            'GlobalAvgPool1D': (),
            'Flatten': (),
            'Reshape': ('target_shape',),
            'Dense': ('units', 'activation', 'initializer'),
            'BatchNorm': ('momentum', 'epsilon'),
            'LayerNorm': ('epsilon',),
            'Dropout': ('rate',),
            'SkipConnection': ('skip_from', 'operation'),
            'ZeroPadding2D': (),
            'Upsample2D': ('scale_factor', 'mode'),
            'Embedding': ('vocab_size', 'embed_dim'),
            'Activation': ('activation', 'alpha'),
            'MultiHeadAttention': ('d_model', 'num_heads'),
            'PositionalEncoding': ('max_len',),
        }
        keys = arg_map.get(layer_type, ())
        return {k: getattr(layer, k) for k in keys}

    def save(self, filename):
        model_state = {'layers': [], 'compiled': self.compiled}
        for layer in self.layers:
            layer_state = {
                'type': type(layer).__name__,
                'init_args': self._get_init_args(layer),
                'state': layer.__dict__,
            }
            model_state['layers'].append(layer_state)
        with open(filename, 'wb') as file:
            pickle.dump(model_state, file)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            model_state = pickle.load(file)
        from . import layers as layer_module
        from . import activations as act_module
        from . import attention as attn_module
        new_model = cls()
        new_model.compiled = model_state.get('compiled', False)
        for layer_state in model_state['layers']:
            layer_name = layer_state['type']
            layer_class = (getattr(layer_module, layer_name, None)
                           or getattr(act_module, layer_name, None)
                           or getattr(attn_module, layer_name, None))
            layer_instance = layer_class(**layer_state['init_args'])
            layer_instance.__dict__.update(layer_state['state'])
            new_model.layers.append(layer_instance)
        print(f"Model loaded from {filename}")
        return new_model
