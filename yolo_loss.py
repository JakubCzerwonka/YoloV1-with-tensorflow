import tensorflow as tf


def _xywh_to_x1y1x2y2(boxes):
    """Convert (x_center, y_center, w ,h) to (xmin, ymin, xmax, ymax)"""
    x, y, w, h = tf.split(boxes, 4, axis=-1)
    xmin = x - w / 2.0
    ymin = y - h / 2.0
    xmax = x + w / 2.0
    ymax = y + h / 2.0
    return tf.concat([xmin, ymin, xmax, ymax], axis=-1)

def _iou_xywh(boxes1, boxes2):
    """Counts IoU"""
    b1 = _xywh_to_x1y1x2y2(boxes1)
    b2 = _xywh_to_x1y1x2y2(boxes2)

    b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(b1, 4, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(b2, 4, axis=-1)

    inter_x1 = tf.maximum(b1_x1, b2_x1)
    inter_y1 = tf.maximum(b1_y1, b2_y1)
    inter_x2 = tf.minimum(b1_x2, b2_x2)
    inter_y2 = tf.minimum(b1_y2, b2_y2)

    inter_w = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = tf.maximum(b1_x2 - b1_x1, 0.0) * tf.maximum(b1_y2 - b1_y1, 0.0)
    area2 = tf.maximum(b2_x2 - b2_x1, 0.0) * tf.maximum(b2_y2 - b2_y1, 0.0)

    union = area1 + area2 - inter_area
    iou = tf.where(union > 0.0, inter_area / union, tf.zeros_like(inter_area))
    return tf.squeeze(iou, axis=-1)


class MyYoloLoss(tf.keras.losses.Loss):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5, name="yolo_loss"):
        super().__init__(name=name)
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)


    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]

        # Get preds.                                                      batch, S, S, B, 5]
        pred_boxes_list = []
        for i in range(self.B):
            s = i * 5
            pred_boxes_list.append(y_pred[..., s:s+5])
        pred_boxes = tf.stack(pred_boxes_list, axis=-2)                     # [batch,S,S,B,5]

        pred_coords = pred_boxes[..., 0:4]                                  # [batch,S,S,B,4]
        pred_confs  = pred_boxes[..., 4:5]                                  # [batch,S,S,B,1]

        true_box = y_true[..., 0:4]                                         # [batch,S,S,4]
        true_box_exp = tf.expand_dims(true_box, axis=-2)                    # [batch,S,S,1,4]

        # Best box by IoU
        ious = _iou_xywh(true_box_exp, pred_coords)                         # [batch,S,S,B]
        resp_idx = tf.argmax(ious, axis=-1, output_type=tf.int32)           # [batch,S,S]
        resp_onehot = tf.one_hot(resp_idx, depth=self.B, dtype=tf.float32)  # [batch,S,S,B]
        resp_onehot = tf.expand_dims(resp_onehot, axis=-1)                  # [batch,S,S,B,1]
        chosen_box = tf.reduce_sum(pred_boxes * resp_onehot, axis=-2)       # [batch,S,S,5]

        # If object and if not object mask
        obj_mask = tf.expand_dims(y_true[..., 4], axis=-1)                  # [batch,S,S,1]
        obj_mask_float = tf.cast(obj_mask, tf.float32)
        obj_mask_expanded = tf.expand_dims(obj_mask_float, axis=-2)         # [batch,S,S,1,1]
        noobj_mask_expanded = 1.0 - obj_mask_expanded

        # x, y, w, h loss part
        xy_true = y_true[..., 0:2]
        xy_pred = chosen_box[..., 0:2]
        xy_loss = tf.reduce_sum(tf.square(xy_true - xy_pred), axis=-1, keepdims=True)
        xy_loss = tf.reduce_sum(xy_loss * obj_mask_float)

        wh_true = y_true[..., 2:4]
        wh_pred = chosen_box[..., 2:4]
        wh_pred_pos = tf.maximum(wh_pred, 0.0)
        wh_true_pos = tf.maximum(wh_true, 0.0)
        wh_loss = tf.square(tf.sqrt(tf.clip_by_value(wh_true_pos, 1e-6, 1e6)) - tf.sqrt(tf.clip_by_value(wh_pred_pos, 1e-6, 1e6)))
        wh_loss = tf.reduce_sum(wh_loss, axis=-1, keepdims=True)
        wh_loss = tf.reduce_sum(wh_loss * obj_mask_float)
        coord_loss = self.lambda_coord * (xy_loss + wh_loss)

        # confidence losses
        conf_pred_resp = chosen_box[..., 4:5]
        conf_loss_obj = tf.square(1.0 - conf_pred_resp) * obj_mask_float
        conf_loss_obj = tf.reduce_sum(conf_loss_obj)

        nonresp_in_obj_mask = obj_mask_expanded * (1.0 - resp_onehot)  # [batch,S,S,B,1]
        noobj_total_mask = noobj_mask_expanded + nonresp_in_obj_mask
        conf_all = pred_confs
        conf_loss_noobj = tf.square(conf_all) * noobj_total_mask
        conf_loss_noobj = tf.reduce_sum(conf_loss_noobj)
        conf_loss_noobj = self.lambda_noobj * conf_loss_noobj

        # class loss (MSE on softmax)
        class_true = y_true[..., -self.C:]
        class_logits = y_pred[..., -self.C:]
        # class_prob = tf.nn.softmax(class_logits, axis=-1)
        # class_loss = tf.square(class_true - class_prob)
        class_loss = tf.square(class_true - class_logits)
        class_loss = tf.reduce_sum(class_loss, axis=-1, keepdims=True)
        class_loss = tf.reduce_sum(class_loss * obj_mask_float)

        # If run_eagerly in model.compile
        # tf.print(
        #     "coord_loss:", coord_loss, "\n", 
        #     "conf_obj:", conf_loss_obj, "\n", 
        #     "conf_noobj:", conf_loss_noobj, "\n", 
        #     "class_loss:", class_loss, "\n",
        # )

        total = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        total = total / tf.cast(batch_size, tf.float32)

        return total