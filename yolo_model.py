import tensorflow as tf

def yolo_v1_model(img_size=(448, 448, 3), S=7, B=2, C=20):
    """Creates a slightly modified YOLO v1 model"""
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=img_size
    )

    # Freezing weights
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    x = tf.keras.layers.Conv2D(1024, (2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(2048, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output_dim = S * S * (B * 5 + C)
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.Reshape((S, S, B * 5 + C))(x)

    xy = tf.keras.layers.Lambda(lambda t: tf.sigmoid(t[..., 0:2*B]))(x)
    wh = tf.keras.layers.Lambda(lambda t: t[..., 2*B:4*B])(x)
    conf = tf.keras.layers.Lambda(lambda t: tf.sigmoid(t[..., 4*B:5*B]))(x)
    classes = tf.keras.layers.Lambda(lambda t: tf.nn.softmax(t[..., 5*B:], axis=-1))(x)

    output = tf.keras.layers.Concatenate(axis=-1)([xy, wh, conf, classes])

    model = tf.keras.Model(inputs=base_model.input, outputs=output, name="YOLOv1")

    return model
