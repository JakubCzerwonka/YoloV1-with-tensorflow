import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import math
from yolo_loss import YoloLoss
from yolo_model import yolo_v1_model
from yolo_preprocessing import *

img_size = (350, 350)
S, B, C = 7, 2, 20

PATH_DATASET_TRAIN = "./PASCALVOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"


files = [os.path.join(PATH_DATASET_TRAIN, f) for f in sorted(os.listdir(PATH_DATASET_TRAIN)) if f.endswith(".xml")]
train_files, valid_files = train_test_split(files, test_size=0.2, random_state=42, shuffle=True)

# Removing classes head, hand, foot like in orginal paper
class_names = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor", "head", "foot", "hand"]
remove_these_vals = ['head', 'hand', 'foot']

batch_size = 16
train_gen = DataGenerator(train_files, img_size, class_names, remove_these_vals, S=S, B=B, C=C, batch_size=batch_size, shuffle=True, augment=True)
valid_gen = DataGenerator(valid_files, img_size, class_names, remove_these_vals, S=S, B=B, C=C, batch_size=1, shuffle=False, augment=False)

checkPointCallback = tf.keras.callbacks.ModelCheckpoint(
    './model_trained/model.keras',
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    initial_value_threshold=None,
)

model = yolo_v1_model(img_size=(img_size[0], img_size[1], 3), S=S, B=B, C=C)

epochs = 250
loss_fn = YoloLoss(S=S, B=B, C=C, lambda_coord=5.0, lambda_noobj=0.5)
steps = math.ceil(len(train_files) / batch_size) * epochs
lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-4, steps, t_mul=1.0, m_mul=1.0, alpha=1e-3)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=5e-4)

model.compile(loss=loss_fn, optimizer=optimizer)

model.fit(train_gen, 
          validation_data=valid_gen, 
          epochs=epochs, 
          batch_size=1, 
          shuffle=True, 
          callbacks=[checkPointCallback])