from typing import List
import collections

import cv2
import numpy as np


BBox = collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def coco80_to_coco91_class(i):  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x[i]


def yolov7(
        img_det,
        interpreter_detect,
        nms_threshold,
    ) -> List[Object]:

    def make(i):
        c_x, c_y, w, h = bboxes[i]
        xmin = c_x - w / 2
        ymin = c_y - h / 2
        xmax = c_x + w / 2
        ymax = c_y + h / 2

        return Object(
            id=coco80_to_coco91_class(classes[i])-1,
            score=scores[i],
            bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    input_details = interpreter_detect.get_input_details()
    output_details = interpreter_detect.get_output_details()

    # quantize
    scale_i8, shift_i8 = input_details[0]['quantization']
    img_det = np.array(img_det, dtype=np.float32) / 255.
    img_det = np.array(img_det / scale_i8 + shift_i8, dtype=np.int8)
    img_det = img_det[None, ...] # add batch dim

    # inference
    interpreter_detect.set_tensor(input_details[0]['index'], img_det)
    interpreter_detect.invoke()
    output_data = interpreter_detect.get_tensor(output_details[0]['index'])

    # dequantize
    scale_i8, shift_i8 = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - shift_i8) * scale_i8
    bboxes = output_data[0, :, :4]
    confs = output_data[0, :, 4]
    scores = output_data[0, :, 5:]

    # get class, denormalize xywh
    classes = np.argmax(scores, axis=1)
    scores = scores[np.arange(scores.shape[0]), classes] # one class score
    scores *= confs # yolov7 specific
    bboxes[:, 0::2] = bboxes[:, 0::2] * img_det.shape[2] # xw
    bboxes[:, 1::2] = bboxes[:, 1::2] * img_det.shape[1] # yh
    bboxes = bboxes.astype(np.int32)

    # calculate nmsboxes (agnostic)
    max_wh = np.array([img_det.shape[1], img_det.shape[2], 0, 0], dtype=np.int32)
    nmsbox = bboxes + classes.astype(np.int32)[:, None] * max_wh[None, :]

    # NMS
    idx = cv2.dnn.NMSBoxes(nmsbox, scores, nms_threshold, 0.45)
    confs = confs[idx]
    bboxes = bboxes[idx]
    scores = scores[idx]
    classes = classes[idx]

    objs = [make(i) for i in range(len(idx))]

    return objs

def yolov8(
        img_det,
        interpreter_detect,
        nms_threshold,
    ) -> List[Object]:
    def make(i):
        c_x, c_y, w, h = bboxes[i]
        xmin = c_x - w / 2
        ymin = c_y - h / 2
        xmax = c_x + w / 2
        ymax = c_y + h / 2
        return Object(
            id=coco80_to_coco91_class(classes[i])-1,
            score=scores[i],
            bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    input_details = interpreter_detect.get_input_details()
    output_details = interpreter_detect.get_output_details()

    # quantize
    scale_i8, shift_i8 = input_details[0]['quantization']
    img_det = np.array(img_det, dtype=np.float32) / 255.
    img_det = np.array(img_det / scale_i8 + shift_i8, dtype=np.int8)
    img_det = img_det[None, ...] # add batch dim

    # inference
    interpreter_detect.set_tensor(input_details[0]['index'], img_det)
    interpreter_detect.invoke()
    output_data = interpreter_detect.get_tensor(output_details[0]['index'])

    # dequantize
    scale_i8, shift_i8 = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - shift_i8) * scale_i8
    output_data = np.transpose(output_data[0, ...]) # CV2 compatiblity
    bboxes = output_data[:, :4]
    scores = output_data[:, 4:]

    # get class, denormalize xywh
    classes = np.argmax(scores, axis=1)
    scores = scores[np.arange(scores.shape[0]), classes] # one class score
    bboxes[:, 0::2] = bboxes[:, 0::2] * img_det.shape[2] # xw
    bboxes[:, 1::2] = bboxes[:, 1::2] * img_det.shape[1] # yh
    bboxes = bboxes.astype(np.int32)

    # calculate nmsboxes (agnostic)
    max_wh = np.array([img_det.shape[1], img_det.shape[2], 0, 0], dtype=np.int32)
    nmsbox = bboxes + classes.astype(np.int32)[:, None] * max_wh[None, :]

    # NMS
    idx = cv2.dnn.NMSBoxes(nmsbox, scores, nms_threshold, 0.45)
    bboxes = bboxes[idx]
    scores = scores[idx]
    classes = classes[idx]

    objs = [make(i) for i in range(len(idx))]

    return objs

def yolo26(
        img_det,
        interpreter_detect,
        score_threshold,
        ratio,
        pad
    ) -> List[Object]:

    input_details = interpreter_detect.get_input_details()
    output_details = interpreter_detect.get_output_details()

    # quantize
    scale_i8, shift_i8 = input_details[0]['quantization']
    img_det_float = np.array(img_det, dtype=np.float32) / 255.
    img_det_quant = np.array(img_det_float / scale_i8 + shift_i8, dtype=np.int8)
    img_det_quant = img_det_quant[None, ...] # add batch dim

    # inference
    interpreter_detect.set_tensor(input_details[0]['index'], img_det_quant)
    interpreter_detect.invoke()

    # The model card says two outputs: boxes and scores.
    # We determine which is which by checking the shape.
    if output_details[0]['shape'][1] == 4: # boxes
        boxes_tensor_index = output_details[0]['index']
        scores_tensor_index = output_details[1]['index']
        boxes_quant_params = output_details[0]['quantization']
        scores_quant_params = output_details[1]['quantization']
    else: # scores
        boxes_tensor_index = output_details[1]['index']
        scores_tensor_index = output_details[0]['index']
        boxes_quant_params = output_details[1]['quantization']
        scores_quant_params = output_details[0]['quantization']

    boxes_data = interpreter_detect.get_tensor(boxes_tensor_index)
    scores_data = interpreter_detect.get_tensor(scores_tensor_index)

    # dequantize
    boxes_scale, boxes_zp = boxes_quant_params
    scores_scale, scores_zp = scores_quant_params
    
    boxes = (boxes_data.astype(np.float32) - boxes_zp) * boxes_scale
    scores = (scores_data.astype(np.float32) - scores_zp) * scores_scale

    boxes = np.squeeze(boxes).T # (8400, 4)
    scores = np.squeeze(scores).T # (8400, 80)
    
    classes = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    
    selected_indices = np.where(max_scores > score_threshold)[0]
    
    if len(selected_indices) == 0:
        return []
        
    final_boxes = boxes[selected_indices] # normalized cxcywh
    final_scores = max_scores[selected_indices]
    final_classes = classes[selected_indices]
    
    # Denormalize to letterboxed image size
    h, w = img_det.shape[:2]
    final_boxes[:, 0::2] *= w # cx, w
    final_boxes[:, 1::2] *= h # cy, h

    # Convert to xmin, ymin, xmax, ymax
    final_boxes[:, 0] -= final_boxes[:, 2] / 2 # xmin
    final_boxes[:, 1] -= final_boxes[:, 3] / 2 # ymin
    final_boxes[:, 0] += final_boxes[:, 2] # xmax
    final_boxes[:, 1] += final_boxes[:, 3] # ymax

    # Adjust for padding and scaling from letterbox
    final_boxes[:, 0::2] -= pad[0]  # x padding
    final_boxes[:, 1::2] -= pad[1]  # y padding
    final_boxes /= ratio

    return [Object(id=coco80_to_coco91_class(final_classes[i])-1, score=final_scores[i], bbox=BBox(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3])) for i, b in enumerate(final_boxes)]