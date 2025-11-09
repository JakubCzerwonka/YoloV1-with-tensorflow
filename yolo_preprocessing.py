import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
import albumentations as A


def encode_yolo_targets(y_box, y_label, S, B, C):
    """Encodes target to YOLOv1 target tensor"""
    target = np.zeros((S, S, B * 5 + C), dtype=np.float32)
    y_box = np.array(y_box, dtype=np.float32)
    y_label = np.array(y_label, dtype=np.int32)

    num_objs = y_box.shape[0]
    for obj_idx in range(num_objs):
        x, y, w, h = y_box[obj_idx].reshape(-1)

        cls = int(y_label)
        
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
    y_new_labels = []

    for i in range(len(y_labels)):
        single_img_labels = []
        for j in range(len(y_labels[i])):
            label = y_labels[i][j]
            label = class_names.index(label)
            single_img_labels.append(j)
        y_new_labels.append(label)
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
    """Load and preprocess data from xml files"""
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


def load_dataset(PATH, remove_list, image_target_size=(448, 448), n_img=None):
    """Load dataset"""
    X, y_box, y_label = [], [], []
    xml_files_list = sorted(os.listdir(PATH + "Annotations/"))

    if n_img is not None:
        xml_files_list = xml_files_list[:n_img]

    for xml_file in xml_files_list:
        img = Image.open(PATH + "JPEGImages/" + xml_file[:-3] + "jpg")

        org_w, org_h = img.size
        img = img.resize(image_target_size)
        new_w, new_h = img.size

        # To scales boxes
        scaler_x, scaler_y = new_w / org_w, new_h / org_h

        xml_path = PATH + "Annotations/" + xml_file
        boxes, labels = preprocess_xml(xml_path, scaler_x, scaler_y, remove_list)

        X.append(img)
        y_box.append(boxes)
        y_label.append(labels)

    return np.array(X), y_box, y_label


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator with data augmentation"""
    def __init__(self, file_list, img_target_size, class_names, remove_these_vals, S, B, C, batch_size=32, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.file_list = file_list
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.img_target_size = img_target_size
        self.class_names = class_names
        self.remove_those_vals = remove_these_vals
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.4
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
                A.Blur(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ],
            bbox_params=A.BboxParams(
                format='coco',
                label_fields=['class_labels']
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
