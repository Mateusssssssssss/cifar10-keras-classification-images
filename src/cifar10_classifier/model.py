"""CNNs com pré-processamento incorporado ao modelo."""


def residual_features(x, weight_decay, layers, regularizers):
    regularizer = regularizers.l2(weight_decay)

    def convolution(tensor, filters, size, stride=1):
        tensor = layers.Conv2D(
            filters, size, strides=stride, padding="same", use_bias=False,
            kernel_regularizer=regularizer,
        )(tensor)
        return layers.BatchNormalization()(tensor)

    x = layers.Activation("relu")(convolution(x, 32, 3))
    x = layers.MaxPooling2D()(x)
    for filters, stride in ((32, 1), (64, 2), (128, 2), (128, 1)):
        shortcut = x
        if stride == 2:
            shortcut = convolution(shortcut, filters, 1, stride)
        x = layers.Activation("relu")(convolution(x, filters, 3, stride))
        x = convolution(x, filters, 3)
        x = layers.Activation("relu")(layers.Add()([x, shortcut]))
    return x


def build_cnn(
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    *,
    filters_per_block: tuple[int, ...] = (32, 64, 128),
    architecture: str = "baseline",
):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(32, 32, 3), dtype="float32", name="image_rgb")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.RandomFlip("horizontal", seed=42)(x)
    x = layers.RandomTranslation(0.08, 0.08, seed=42)(x)
    if architecture == "residual":
        x = residual_features(x, weight_decay, layers, keras.regularizers)
    elif architecture == "baseline":
        for filters in filters_per_block:
            x = layers.Conv2D(
                filters, 3, padding="same", use_bias=False,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(dropout / 2)(x)
    else:
        raise ValueError(f"Arquitetura desconhecida: {architecture}")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation="softmax", name="class_probabilities")(x)
    model_name = "cifar10_residual_cnn" if architecture == "residual" else "cifar10_cnn"
    model = keras.Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
