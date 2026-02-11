import tensorflow as tf


def yolo_v1_model(img_size=(448, 448, 3), S=7, B=2, C=20):
    """Creates a slightly modified YOLO v1 model"""
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=img_size
    )

    x = base_model.output

    x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.ReLU()(x)

    output_dim = S * S * (B * 5 + C)
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.Reshape((S, S, B * 5 + C))(x)

    xy = tf.keras.layers.Activation("linear")(x[..., 0:2*B])
    wh = x[..., 2*B:4*B]

    conf = tf.keras.layers.Activation("linear")(x[..., 4*B:5*B])
    classes = tf.keras.layers.Softmax(axis=-1)(x[..., 5*B:])

    output = tf.keras.layers.Concatenate(axis=-1)([xy, wh, conf, classes])

    model = tf.keras.Model(inputs=base_model.input, outputs=output, name="YOLOv1")

    return model