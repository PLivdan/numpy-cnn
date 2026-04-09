from .nn import (
    Model, Conv2D, Pooling2D, GlobalAvgPool2D, GlobalAvgPool1D,
    Flatten, Dense, BatchNorm, LayerNorm, Dropout, SkipConnection, Embedding,
    Activation, LSTM, GRU, Bidirectional, SeparableConv2D,
    MultiHeadAttention, PositionalEncoding, CausalMultiHeadAttention,
    DilatedConv2D, AdaptiveAvgPool2D, SEBlock, FeedForward,
    TransformerEncoderBlock, SpatialDropout, DropPath,
)


def LeNet5(num_classes=10, input_shape=(None, 28, 28, 1)):
    m = Model()
    m.add(Conv2D(6, (5, 5), activation='relu', padding=0))
    m.add(Pooling2D((2, 2), stride=2))
    m.add(Conv2D(16, (5, 5), activation='relu', padding=0))
    m.add(Pooling2D((2, 2), stride=2))
    m.add(Flatten())
    m.add(Dense(120, activation='relu'))
    m.add(Dense(84, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def SimpleCNN(num_classes=10, input_shape=(None, 28, 28, 1), filters=[32, 64]):
    m = Model()
    for f in filters:
        m.add(Conv2D(f, (3, 3), activation='relu', padding='same'))
        m.add(BatchNorm())
        m.add(Pooling2D((2, 2), stride=2))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def ResNetBlock(model, filters, downsample=False, block_start_idx=None):
    stride = 2 if downsample else 1
    padding = 'same' if not downsample else 0
    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(BatchNorm())
    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(BatchNorm())
    if block_start_idx is not None:
        model.add(SkipConnection(skip_from=block_start_idx))


def ResNet(num_classes=10, input_shape=(None, 28, 28, 1)):
    m = Model()
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    start = len(m.layers) - 1
    ResNetBlock(m, 32, block_start_idx=start)
    m.add(Pooling2D((2, 2), stride=2))
    start = len(m.layers) - 1
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    m.add(SkipConnection(skip_from=start))
    m.add(GlobalAvgPool2D())
    m.add(Dense(num_classes, activation='softmax'))
    return m


def MobileNet(num_classes=10, input_shape=(None, 28, 28, 1)):
    pass  # SeparableConv2D already imported at top
    m = Model()
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    for f in [64, 128, 128]:
        m.add(SeparableConv2D(f, (3, 3), padding='same', activation='relu'))
        m.add(BatchNorm())
    m.add(Pooling2D((2, 2), stride=2))
    for f in [256, 256]:
        m.add(SeparableConv2D(f, (3, 3), padding='same', activation='relu'))
        m.add(BatchNorm())
    m.add(GlobalAvgPool2D())
    m.add(Dropout(0.2))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def VGGStyle(num_classes=10, input_shape=(None, 28, 28, 1)):
    m = Model()
    for f in [32, 32]:
        m.add(Conv2D(f, (3, 3), activation='relu', padding='same'))
    m.add(Pooling2D((2, 2), stride=2))
    for f in [64, 64]:
        m.add(Conv2D(f, (3, 3), activation='relu', padding='same'))
    m.add(Pooling2D((2, 2), stride=2))
    m.add(Flatten())
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def LSTMClassifier(vocab_size, embed_dim, hidden_size, num_classes, max_len=100):
    m = Model()
    m.add(Embedding(vocab_size, embed_dim))
    m.add(LSTM(hidden_size))
    m.add(Dropout(0.3))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def BiLSTMClassifier(vocab_size, embed_dim, hidden_size, num_classes, max_len=100):
    m = Model()
    m.add(Embedding(vocab_size, embed_dim))
    m.add(Bidirectional(LSTM(hidden_size)))
    m.add(Dropout(0.3))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def GRUClassifier(vocab_size, embed_dim, hidden_size, num_classes, max_len=100):
    m = Model()
    m.add(Embedding(vocab_size, embed_dim))
    m.add(GRU(hidden_size))
    m.add(Dropout(0.3))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def TransformerClassifier(d_model, num_heads, num_layers, num_classes,
                          max_len=100, d_ff=None, dropout=0.1):
    m = Model()
    m.add(PositionalEncoding(max_len=max_len))
    for _ in range(num_layers):
        m.add(TransformerEncoderBlock(d_model, num_heads, d_ff=d_ff, dropout_rate=dropout))
    m.add(GlobalAvgPool1D())
    m.add(Dropout(dropout))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def Autoencoder(latent_dim=32, input_shape=(None, 28, 28, 1)):
    from .nn.layers import ConvTranspose2D, Reshape
    m = Model()
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(Pooling2D((2, 2), stride=2))
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(Pooling2D((2, 2), stride=2))
    m.add(Flatten())
    m.add(Dense(latent_dim, activation='relu'))
    m.add(Dense(7 * 7 * 64, activation='relu'))
    m.add(Reshape((7, 7, 64)))
    m.add(ConvTranspose2D(32, (3, 3), stride=2, activation='relu'))
    m.add(ConvTranspose2D(1, (3, 3), stride=2, activation='sigmoid'))
    return m


def SEResNet(num_classes=10, input_shape=(None, 28, 28, 1)):
    m = Model()
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    m.add(SEBlock(reduction=8))
    m.add(Pooling2D((2, 2), stride=2))
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(BatchNorm())
    m.add(SEBlock(reduction=8))
    m.add(GlobalAvgPool2D())
    m.add(Dense(num_classes, activation='softmax'))
    return m
