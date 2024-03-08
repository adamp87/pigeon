from typing import List

import cv2
import numpy as np
from pycoral.adapters.detect import BBox, Object


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
    bboxes[:, 0::2] = bboxes[:, 0::2] * img_det.shape[1] # xw
    bboxes[:, 1::2] = bboxes[:, 1::2] * img_det.shape[2] # yh
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
    bboxes[:, 0::2] = bboxes[:, 0::2] * img_det.shape[1] # xw
    bboxes[:, 1::2] = bboxes[:, 1::2] * img_det.shape[2] # yh
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