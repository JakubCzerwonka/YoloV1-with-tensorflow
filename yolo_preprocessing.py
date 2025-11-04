import tensorflow as tf
import numpy as np
import os
from PIL import Image
import xml.etree.ElementTree as ET
from IPython.display import clear_output


class PreprocessingClass:
    def __init__(self, S, B, C):
        super().__init__()
        self.S = S 
        self.B = B
        self.C = C

    def load_dataset(PATH, image_target_size, scale_bbox=True, n_imgs=1, full_imgs=False):
        X, y_bbox, y_label = [], [], []
        xml_files_list = sorted(os.listdir(PATH + "Annotations/"))
        file_count = 0

        if full_imgs == True:
            n_imgs = len(xml_files_list)

        for xml_file in xml_files_list:
            clear_output(wait=True)
            if file_count <= n_imgs - 1:
                
                img = Image.open(PATH + "JPEGImages/" + xml_file[:-3] + "jpg")

                if scale_bbox == True:
                    org_w, org_h = img.size

                    img = img.resize(image_target_size)
                    new_w, new_h = img.size

                    scaler_x, scaler_y = new_w / org_w, new_h / org_h
                if scale_bbox == False:
                    scaler_x, scaler_y = 1, 1

                tree = ET.parse(PATH + "Annotations/" + xml_file)
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
                
                y_bbox_img = []
                for i in range(len(xmins)):
                    # x, y = (xmins[i] + xmaxs[i]) / 2.0, (ymins[i] + ymaxs[i])
                    # w, h = xmaxs[i] - xmins[i], ymaxs[i] - ymins[i]
                    x, y = xmins[i], ymins[i]
                    w, h = xmaxs[i] - x, ymaxs[i] - y
                    y_bbox_img.append([x, y, w , h])

                X.append(img)
                y_bbox.append(y_bbox_img)
                y_label.append(labels)
                file_count += 1
                print(f"Loaded file: {file_count}/{n_imgs}")

        if scale_bbox == True:
            return np.array(X), y_bbox, y_label
        if scale_bbox == False:
            return X, y_bbox, y_label


    def to_one_hot(y_labels):
        y_new_labels = []
        uniq_classes = []
        uniq_counts = []

        for i in range(len(y_labels)):
            for j in range(len(y_labels[i])):
                if y_labels[i][j] not in uniq_classes:
                    uniq_classes.append(y_labels[i][j])
        
        for i in range(len(y_labels)):
            single_img_labels = []
            for j in range(len(y_labels[i])):
                label = uniq_classes.index(y_labels[i][j])
                uniq_counts.append(label)
                single_img_labels.append(label)
            y_new_labels.append(single_img_labels)

        _, counts = np.unique_counts(uniq_counts)
        return y_new_labels, uniq_classes, counts


    def xywh_pixels_to_normalized_centers_per_image(bboxes_list, image_sizes):
        
        """
        (x, y, w, h) -> (cx, cy, w, h), and normalize to [0, 1]
        """
        W, H = image_sizes[0], image_sizes[1]
        all_normalized = []
        for i in range(len(bboxes_list)):
            out = []
            arr_1 = np.array(bboxes_list[i], dtype=np.float32)
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


    def encode_yolo_targets(y_bbox, y_label, S, B, C, *,
                            use_sqrt_wh=False):
        """
        Encodes bboxes and labels to YOLOv1 target tensor
        """
        
        target = np.zeros((S, S, B * 5 + C), dtype=np.float32)
        y_bbox = np.array(y_bbox, dtype=np.float32)
        y_label = np.array(y_label, dtype=np.int32)

        num_objs = y_bbox.shape[0]
        for obj_idx in range(num_objs):
            x, y, w, h = y_bbox[obj_idx]
            cls = int(y_label[obj_idx])

            i = min(int(x * S), S - 1)
            j = min(int(y * S), S - 1)

            conf_vals = [target[j, i, b * 5 + 4] for b in range(B)]
            if any(conf_vals):  # if cell is not empty
                    continue 

            # local middle cords in cell
            x_cell = x * S - i
            y_cell = y * S - j

            if use_sqrt_wh:
                w_val = np.sqrt(max(w, 0.0))
                h_val = np.sqrt(max(h, 0.0))
            else:
                w_val = w
                h_val = h

            # Save gt with conf. val. to first 5 values
            target[j, i, 0:5] = np.array([x_cell, y_cell, w_val, h_val, 1.0], dtype=np.float32)

            # one hot
            target[j, i, B * 5 + cls] = 1.0

        return target


    def encode_batch(y_bbox_list, y_label_list, S, B, C):
        data_len = len(y_bbox_list)
        targets = np.zeros((data_len, S, S, B * 5 + C), dtype=np.float32)

        for idx in range(data_len):
            bboxes = np.array(y_bbox_list[idx], dtype=np.float32)
            labels = np.array(y_label_list[idx], dtype=np.int32)
            targets[idx] = PreprocessingClass.encode_yolo_targets(bboxes, labels, S, B, C)
        return targets
    
    
    def decode_yolo_predictions(pred, S, B, C, img_size, conf_thresh=0.5, use_sqrt_wh=False):
        W, H = img_size
        boxes = []

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

                    if use_sqrt_wh:
                        w = w ** 2
                        h = h ** 2

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

                    boxes.append([xmin, ymin, w, h, score, cls_id])

        return boxes


    def decode_batch(preds_map, S, B, C, img_sizes, conf_thresh=0.5):
        all_boxes = []
        for img_idx in range(preds_map.shape[0]):
            boxes_for_curr_img = PreprocessingClass.decode_yolo_predictions(preds_map[img_idx], S=S, B=B, C=C, img_size=img_sizes, conf_thresh=conf_thresh, use_sqrt_wh=False)
            all_boxes.append(boxes_for_curr_img)

        return all_boxes
    
    
    def check_map_values(y_encoded):
        """Prints min, avg, max for encodec data"""
        bboxes = y_encoded[..., :5]
        mask = tf.reduce_any(tf.not_equal(bboxes, 0), axis=-1)

        print(f"x center: min val = {np.min(bboxes[mask][:, 0]):.3f}, mean val = {tf.reduce_mean(bboxes[mask][:, 0]):.3f}, max val = {np.max(bboxes[mask][:, 1]):.3f}")
        print(f"y center: min val = {np.min(bboxes[mask][:, 1]):.3f}, mean val = {tf.reduce_mean(bboxes[mask][:, 1]):.3f}, max val = {np.max(bboxes[mask][:, 2]):.3f}")
        print(f"w center: min val = {np.min(bboxes[mask][:, 2]):.3f}, mean val = {tf.reduce_mean(bboxes[mask][:, 2]):.3f}, max val = {np.max(bboxes[mask][:, 3]):.3f}")
        print(f"h center: min val = {np.min(bboxes[mask][:, 3]):.3f}, mean val = {tf.reduce_mean(bboxes[mask][:, 3]):.3f}, max val = {np.max(bboxes[mask][:, 4]):.3f}")


    def bbox_labels_info(y_bbox):
        xmin_avg, ymin_avg, w_avg, h_avg = 0, 0, 0, 0
        n_bboxes = 0

        xmin_min_val, xmin_max_val = y_bbox[0][0][0], y_bbox[0][0][0]
        ymin_min_val, ymin_max_val = y_bbox[0][0][1], y_bbox[0][0][1]
        w_min_val, w_max_val = y_bbox[0][0][2], y_bbox[0][0][2]
        h_min_val, h_max_val = y_bbox[0][0][3], y_bbox[0][0][3]


        for i in range(len(y_bbox)):
            y_sample_bbox = y_bbox[i]
            for j in range(len(y_sample_bbox)):
                n_bboxes += 1
                if y_sample_bbox[j][0] < xmin_min_val:
                    xmin_min_val = y_sample_bbox[j][0]
                if y_sample_bbox[j][0] > xmin_max_val:
                    xmin_max_val = y_sample_bbox[j][0]

                if y_sample_bbox[j][1] < ymin_min_val:
                    ymin_min_val = y_sample_bbox[j][1]
                if y_sample_bbox[j][1] > ymin_max_val:
                    ymin_max_val = y_sample_bbox[j][1]

                if y_sample_bbox[j][2] < w_min_val:
                    w_min_val = y_sample_bbox[j][2]
                if y_sample_bbox[j][2] > w_max_val:
                    w_max_val = y_sample_bbox[j][2]

                if y_sample_bbox[j][3] < h_min_val:
                    h_min_val = y_sample_bbox[j][3]
                if y_sample_bbox[j][3] > h_max_val:
                    h_max_val = y_sample_bbox[j][3]

                xmin_avg += y_sample_bbox[j][0]
                ymin_avg += y_sample_bbox[j][1]
                w_avg += y_sample_bbox[j][2]
                h_avg += y_sample_bbox[j][3]

        xmin_avg /= n_bboxes
        ymin_avg /= n_bboxes
        w_avg /= n_bboxes
        h_avg /= n_bboxes

        print(f"xmin_min_val: {xmin_min_val:.4}, xmin_avg: {xmin_avg:.4}, xmin_max_val: {xmin_max_val:.4}")
        print(f"ymin_min_val: {ymin_min_val:.4}, ymin_avg: {ymin_avg:.4}, ymin_max_val: {ymin_max_val:.4}")
        print(f"w_min_val: {w_min_val:.4}, w_avg: {w_avg:.4}, w_max_val: {w_max_val:.4}")
        print(f"h_min_val: {h_min_val:.4}, h_avg: {h_avg:.4}, h_max_val: {h_max_val:.4}")
        print(f"Number of bboxes: {n_bboxes}")