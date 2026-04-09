# numpy-cnn

A deep learning framework built entirely with NumPy. No PyTorch, no TensorFlow, no JAX — just NumPy.

Supports CNNs, RNNs, Transformers, and everything needed to train them.

## Install

```bash
git clone https://github.com/PLivdan/numpy-cnn.git
cd numpy-cnn
pip install numpy matplotlib
```

## Quick Start

```python
from numpycnn import *

# Load data
(X_train, y_train), (X_test, y_test) = load_fashion_mnist()
y_train, y_test = one_hot(y_train), one_hot(y_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

# Build model
model = Model()
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(BatchNorm())
model.add(Pooling2D((2,2), stride=2))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Pooling2D((2,2), stride=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(
    input_shape=(None, 28, 28, 1),
    optimizer=Adam(use_agc=True),
    initializer='he'
)
model.summary()

# Train
history = model.fit(
    X_train, y_train, X_val, y_val,
    batch_size=128, epochs=10,
    loss_fn=categorical_crossentropy,
    lr_scheduler=CosineAnnealingLR(initial_lr=0.001, T_max=10),
    callbacks=[EarlyStopping(patience=3)],
)

# Evaluate
preds = model.predict(X_test)
classification_report(y_test, preds)
```

## What's Inside

### Layers
| Layer | Description |
|---|---|
| `Conv2D` | 2D convolution with im2col (stride tricks) |
| `Conv1D` | 1D convolution for sequences |
| `ConvTranspose2D` | Transposed convolution (decoders, U-Net, GANs) |
| `DepthwiseConv2D` | Depthwise convolution |
| `SeparableConv2D` | Depthwise separable convolution (MobileNet) |
| `Dense` | Fully connected layer |
| `RNN` | Vanilla recurrent layer |
| `LSTM` | Long Short-Term Memory |
| `GRU` | Gated Recurrent Unit |
| `Bidirectional` | Wraps any RNN/LSTM/GRU for bidirectional processing |
| `MultiHeadAttention` | Scaled dot-product multi-head attention |
| `PositionalEncoding` | Sinusoidal positional encoding |
| `Embedding` | Token embedding lookup table |
| `BatchNorm` | Batch normalization |
| `LayerNorm` | Layer normalization |
| `Dropout` | Inverted dropout |
| `Pooling2D` / `Pooling1D` | Max/average pooling |
| `GlobalAvgPool2D` / `GlobalAvgPool1D` | Global average pooling |
| `Flatten` / `Reshape` | Shape manipulation |
| `Upsample2D` | Nearest/bilinear upsampling |
| `ZeroPadding2D` | Explicit zero padding |
| `SkipConnection` | Residual connections |
| `Activation` | ReLU, LeakyReLU, ELU, SELU, GELU, SiLU, Mish, Sigmoid, Tanh |

### Optimizers
SGD, SGD+Momentum, RMSprop, AdaGrad, AdaDelta, Adam, AdamW, NAG, AMSGrad, Yogi, AdaFactor

All support adaptive gradient clipping (`use_agc=True`), gradient norm clipping (`clip_norm=1.0`), and value clipping (`clip_value=0.5`).

### Losses
`categorical_crossentropy`, `binary_crossentropy`, `mse_loss`, `huber_loss`, `focal_loss`, `label_smoothing_crossentropy`, `kl_divergence`, `hinge_loss`

### Schedulers
`LRScheduler` (step/patience), `CosineAnnealingLR`, `WarmupScheduler`, `ExponentialLR`

### Training
- `EarlyStopping` with best weight restoration
- `ModelCheckpoint` for saving best models
- `model.freeze()` / `model.unfreeze()` for transfer learning
- `model.save()` / `Model.load()` for persistence
- `DataLoader` for batched iteration
- `ImageDataAugmentor` with rotate, flip, crop, noise, jitter, etc.

### Metrics
`accuracy`, `precision`, `recall`, `f1_score`, `top_k_accuracy`, `confusion_matrix`, `classification_report`

### Datasets
`load_fashion_mnist()`, `load_cifar10()` — downloaded automatically, no keras/tensorflow needed.

## Examples

See [`demo.ipynb`](demo.ipynb) for a complete Fashion-MNIST training example with plots.

## Architecture Examples

**CNN (image classification):**
```python
Conv2D → BatchNorm → Pooling2D → Conv2D → Pooling2D → Flatten → Dense → Softmax
```

**RNN (sequence classification):**
```python
Embedding → LSTM → Dense → Softmax
```

**Transformer encoder (sequence classification):**
```python
Embedding → PositionalEncoding → MultiHeadAttention → LayerNorm → Dense → GlobalAvgPool1D → Dense → Softmax
```

**Autoencoder:**
```python
Conv2D → Pooling2D → Conv2D → ConvTranspose2D → ConvTranspose2D
```

**MobileNet-style:**
```python
Conv2D → SeparableConv2D → SeparableConv2D → GlobalAvgPool2D → Dense
```
