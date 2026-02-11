import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
import albumentations as A


def encode_yolo_targets(y_boxes, y_labels, S, B, C):
    """Encodes target to YOLOv1 target tensor"""
    target = np.zeros((S, S, B * 5 + C), dtype=np.float32)
    y_boxes = np.array(y_boxes, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int32)

    num_objs = y_boxes.shape[0]
    for obj_idx in range(num_objs):
        x, y, w, h = y_boxes[obj_idx].reshape(-1)

        cls = int(y_labels[obj_idx])
        
        i = min(int(x * S), S - 1)
        j = min(int(y * S), S - 1)

        conf_vals = [target[j, i, b * 5 + 4] for b in range(B)]
        if any(conf_vals):              # if cell is not empty
                continue 

        # local mid cords in cell
        x_cell = x * S - i
        y_cell = y * S - j

        # Save gt with conf. val. to first 5 values
        target[j, i, 0:5] = np.array([x_cell, y_cell, w, h, 1.0], dtype=np.float32)

        # one hot
        target[j, i, B * 5 + cls] = 1.0

    return target


def encode_batch(y_box_list, y_label_list, S, B, C):
    """Encodes batch of targets to YOLOv1 target tensor"""
    data_len = len(y_box_list)
    targets = np.zeros((data_len, S, S, B * 5 + C), dtype=np.float32)
    for idx in range(data_len):
        boxes = np.array(y_box_list[idx], dtype=np.float32)
        labels = np.array(y_label_list[idx], dtype=np.int32)
        targets[idx] = encode_yolo_targets(boxes, labels, S=S, B=B, C=C)

    return targets


def decode_yolo_predictions(pred, S, B, img_size, conf_thresh):
    """Decodes target to boxes, conf score and class id"""
    W, H = img_size
    target = []

    for j in range(S):
        for i in range(S):
            cell = pred[j, i]
            class_probs = cell[B*5:]
            cls_id = np.argmax(class_probs)
            cls_prob = class_probs[cls_id]

            for b in range(B):
                base = b * 5
                x, y, w, h, conf = cell[base:base+5]
                score = conf * cls_prob

                if score < conf_thresh:
                    continue

                # Reverse yolo codding
                cx = (i + x) / S
                cy = (j + y) / S

                cx_p = cx * W
                cy_p = cy * H
                w_p  = w * W
                h_p  = h * H

                xmin = cx_p - w_p / 2
                ymin = cy_p - h_p / 2
                xmax = cx_p + w_p / 2
                ymax = cy_p + h_p / 2

                w = xmax - xmin
                h = ymax - ymin

                target.append([xmin, ymin, w, h, score, cls_id])

    return target


def decode_batch(preds_map, S, B, img_size, conf_thresh=0.5):
    """Decodes batch of targets to boxes, conf score and class id"""
    all_boxes = []
    for img_idx in range(preds_map.shape[0]):
        boxes_for_curr_img = decode_yolo_predictions(preds_map[img_idx], S=S, B=B, img_size=img_size, conf_thresh=conf_thresh)
        all_boxes.append(boxes_for_curr_img)

    return all_boxes


def to_one_hot(y_labels, class_names):
    """classes to onehot"""
    y_new_labels = []

    for i in range(len(y_labels)):
        single_img_labels = []
        for j in range(len(y_labels[i])):
            label = y_labels[i][j]
            if label in class_names:
                label_idx = class_names.index(label)
                single_img_labels.append(label_idx)
        y_new_labels.append(single_img_labels)
    return y_new_labels


def xywh_pixels_to_normalized_centers_per_image(boxes_list, image_sizes):
    """ (x, y, w, h) -> (cx, cy, w, h), and normalize to [0, 1] """
    W, H = image_sizes[0], image_sizes[1]
    all_normalized = []
    for i in range(len(boxes_list)):
        out = []
        arr_1 = np.array(boxes_list[i], dtype=np.float32)
        for curr_box in arr_1:
            curr_box = curr_box.reshape(1,4)
            
            x = curr_box[:,0]
            y = curr_box[:,1] 
            w = curr_box[:,2] 
            h = curr_box[:,3]

            cx = x + w * 0.5
            cy = y + h * 0.5
            cx_n = cx / float(W)
            cy_n = cy / float(H)
            w_n = w / float(W)
            h_n = h / float(H)
            normalized = np.array([cx_n, cy_n, w_n, h_n]).reshape(-1, 4)
            out.append(normalized)
        all_normalized.append(out)
    return all_normalized


def preprocess_xml(xml_path, scaler_x, scaler_y, remove_these_vals):
    """Loads and preprocess data from xml files"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xmins = root.findall('.//xmin')
    xmins = [float(xmin.text) * scaler_x for xmin in xmins]

    ymins = root.findall('.//ymin')
    ymins = [float(ymin.text) * scaler_y for ymin in ymins]

    xmaxs = root.findall('.//xmax')
    xmaxs = [float(xmax.text) * scaler_x for xmax in xmaxs]

    ymaxs = root.findall('.//ymax')
    ymaxs = [float(ymax.text) * scaler_y for ymax in ymaxs]

    labels = root.findall('.//name')
    labels = [label.text for label in labels]
    labels = labels[1:] # Skips owner's name

    boxes = []
    for i in range(len(labels)):
        if labels[i] == 'foot' or labels[i] == 'hand' or labels[i] == 'head':
            continue

        x, y = xmins[i], ymins[i]
        w, h = xmaxs[i] - x, ymaxs[i] - y
        boxes.append([x, y, w , h])

    # Remove head, hand and foot classes from labels
    labels = [x for x in labels if x not in remove_these_vals]

    return boxes, labels


def new_load_dataset(file_list, remove_list, image_target_size=(448, 448)):
    """Loads dataset"""
    X, y_box, y_label = [], [], []

    for xml_path in file_list:
        img_path = xml_path[:-22] + "JPEGImages/" + xml_path[-10:-4] + ".jpg"

        img = Image.open(img_path)

        org_w, org_h = img.size
        img = img.resize(image_target_size)
        new_w, new_h = img.size

        # To scales boxes
        scaler_x, scaler_y = new_w / org_w, new_h / org_h

        boxes, labels = preprocess_xml(xml_path, scaler_x, scaler_y, remove_list)

        X.append(img)
        y_box.append(boxes)
        y_label.append(labels)

    return np.array(X), y_box, y_label


def reverse_vgg16(x):
    """Reverse tensorflow.keras.applications.vgg16.preprocess_input"""
    x = np.copy(x)
    x = x[..., ::-1]

    x[..., 0] += 123.68
    x[..., 1] += 116.779
    x[..., 2] += 103.939
    return np.clip(x, 0, 255).astype('uint8')


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator with data augmentation"""
    def __init__(self, file_list, img_target_size, class_names, remove_these_vals, S, B, C, batch_size=32, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.file_list = file_list
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.img_target_size = img_target_size
        self.class_names = class_names
        self.remove_these_vals = remove_these_vals
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment

        if self.augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(350, 350), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.6),
                A.HorizontalFlip(p=0.6),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=20, border_mode=0, p=0.7),
                A.OneOf([A.RandomBrightnessContrast(0.2, 0.2), A.HueSaturationValue(12, 20, 10), A.RGBShift(15, 15, 15)], p=0.7),
                A.OneOf([A.Blur(blur_limit=3), A.MotionBlur(blur_limit=3), A.GaussNoise(var_limit=(10, 30), p=0.25)]),
                A.CoarseDropout(max_holes=6, max_height=0.12, max_width=0.12, p=0.3),
                A.ImageCompression(quality_lower=70, quality_upper=95, p=0.25)
            ],
            bbox_params=A.BboxParams(
                format='coco',
                label_fields=['class_labels'],
                min_visibility=0.4,
                clip=True
            ))
            

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = self._load_batch(batch_files)
        return X, y


    def _load_batch(self, batch_files):
        # Load dataset
        X, y_all_boxes, y_all_labels = [], [], []
        for xml_path in batch_files:
            img_path = xml_path[:-22] + "JPEGImages/" + xml_path[-10:-4] + ".jpg"

            # Img preprocessing
            img = Image.open(img_path)
            org_img_w, org_img_h = img.size 
            img = img.resize(self.img_target_size)
            new_img_w, new_img_h = img.size
            
            # To adjust boxes
            scaler_x, scaler_y = new_img_w / org_img_w, new_img_h / org_img_h

            # Load boxes and labels
            y_boxes, y_labels = preprocess_xml(xml_path, scaler_x, scaler_y, self.remove_these_vals)

            if self.augment and len(y_boxes) > 0:
                transformed = self.transform(
                    image=np.array(img),
                    bboxes=y_boxes,
                    class_labels=y_labels
                )
                img = transformed['image']
                y_boxes = transformed['bboxes']
                y_labels = transformed['class_labels']

            X.append(img)
            y_all_boxes.append(y_boxes)
            y_all_labels.append(y_labels)
        
        # Encode and preprocess to Yolo v1 and VGG16 format
        X = np.array(X)
        X = preprocess_input(X)
        y_all_boxes = xywh_pixels_to_normalized_centers_per_image(y_all_boxes, self.img_target_size)
        y_all_labels = to_one_hot(y_all_labels, self.class_names)
        y = encode_batch(y_all_boxes, y_all_labels, S=self.S, B=self.B, C=self.C)
        
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.file_list)


def nms(y_encoded, S, B, C, iou_thrshold, score_threshold):
    """Preprocessing + Non Maximum Suppresion"""
    batch_size = tf.shape(y_encoded)[0]

    # preprocessing to nms
    bboxes = tf.reshape(y_encoded[..., :5*B], (-1, S, S, B, 5))
    classes = y_encoded[..., 5*B:]

    cords = bboxes[..., :4]
    conf = bboxes[..., 4:5]

    # grid offset
    x_grid = tf.range(S, dtype=tf.float32)
    y_grid = tf.range(S, dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x_grid, y_grid)

    x_grid = tf.reshape(x_grid, (1, S, S, 1, 1))
    y_grid = tf.reshape(y_grid, (1, S, S, 1, 1))

    cx = (cords[..., 0:1] + x_grid) / S
    cy = (cords[..., 1:2] + y_grid) / S
    w = cords[..., 2:3]
    h = cords[..., 3:4]

    # (xc, yc, w, h) -> (xmin, ymin, xmax, ymax)
    ymin = cy - h / 2
    xmin = cx - w / 2
    ymax = cy + h / 2
    xmax = cx + w / 2

    boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    classes = tf.expand_dims(classes, axis=3)
    classes = tf.tile(classes, [1, 1, 1, B, 1])

    scores = conf * classes

    n_bboxes = S * S * B

    boxes = tf.reshape(boxes, (batch_size, n_bboxes, 4))
    scores = tf.reshape(scores, (batch_size, n_bboxes, C))
    boxes = tf.expand_dims(boxes, axis=2)

    nms_output = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=20,
        max_total_size=50,
        iou_threshold=iou_thrshold,
        score_threshold=score_threshold
    )

    return nms_output