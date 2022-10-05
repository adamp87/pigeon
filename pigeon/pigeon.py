import time
import argparse
import logging as log

import cv2
import numpy as np

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.adapters import classify
from pycoral.utils.edgetpu import run_inference
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def pigeon_detected() -> None:
    # TODO: make noise
    log.debug("PIGEON!")


def draw_debug_info(img: np.ndarray, frame_idx: int, proc_time: float) -> None:
    cv2.putText(
        img,
        "frame_idx: %d" % frame_idx,
        (10, 10 + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
        )
    cv2.putText(
        img,
        "proc_time: %s ms" % round(proc_time * 1000, 0),
        (10, 10 + 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Path of video/url to process')
    parser.add_argument('-o', '--output',
                        help='File path for the result image with annotations')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbosity, 0,1,2')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Score threshold for detected birds')
    parser.add_argument('-k', '--top_k', type=int, default=2,
                        help='Max number of classification results')
    parser.add_argument('-t0', '--tpu_0', type=str,
                        help='Device location for detection')
    parser.add_argument('-t1', '--tpu_1', type=str,
                        help='Device location for classification')
    args = parser.parse_args()

    if args.verbose == 1:
        log.root.setLevel(log.INFO)
    elif args.verbose == 2:
        log.root.setLevel(log.DEBUG)

    # use pre-trained models from Coral
    args.labels_coco = r"models/coco_labels.txt"
    args.labels_bird = r"models/inat_bird_labels.txt"
    args.model_pigeon = r"models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
    args.model_detect = r"models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"

    # set up edge devices
    tpu_list = list_edge_tpus()
    if len(tpu_list) == 0:
        log.error("No TPU found")
        exit(-1)
    elif len(tpu_list) == 1:
        args.tpu_0 = ":0"
        args.tpu_1 = ":0"
        log.info("Found 1 TPU, using 1 TPU")
    else:
        if args.tpu_0 is None:
            args.tpu_0 = ":0"
        if args.tpu_1 is None:
            args.tpu_1 = ":1"
        log.info("Found %d TPUs, using 2 TPUs" % len(tpu_list))
    log.info("Detection executed on: %s" % args.tpu_0)
    log.info("Classification executed on: %s" % args.tpu_1)

    # set up edge tpu interpreters
    labels_coco = read_label_file(args.labels_coco)
    labels_bird = read_label_file(args.labels_bird)
    interpreter_detect = make_interpreter(args.model_detect, device=args.tpu_0)
    interpreter_pigeon = make_interpreter(args.model_pigeon, device=args.tpu_1)
    inference_detect_size = common.input_size(interpreter_detect)
    inference_pigeon_size = common.input_size(interpreter_pigeon)
    interpreter_detect.allocate_tensors()
    interpreter_pigeon.allocate_tensors()

    # set up video input
    frame_idx = 0
    cap = cv2.VideoCapture(args.input)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if vid_width == 0 or vid_height == 0:
        ret, frame = cap.read()
        if not ret:
            log.error("Could not open input source")
            exit(-1)
        vid_height, vid_width, _ = frame.shape
    det_scale = (vid_width / inference_detect_size[0], vid_height / inference_detect_size[1])

    # set up video output
    file_writer = None
    if args.output is not None:
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        if cap_fps == 0:
            cap_fps = 25
            log.warning("Could not get FPS from input, set 25 for output")
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        file_writer = cv2.VideoWriter(
            args.output,
            fourcc,
            cap_fps,
            (vid_width, vid_height),
        )

    # process input video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.perf_counter()

        # detect objects in frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # rgb image
        img_det = cv2.resize(img_rgb, inference_detect_size)  # resize for detection
        run_inference(interpreter_detect, img_det.tobytes())  # detection of objects
        objs = detect.get_objects(interpreter_detect, args.threshold)  # get detected object

        for obj in objs:  # for each object
            obj_label = labels_coco.get(obj.id, obj.id)  # get label
            if obj_label != "bird":
                continue  # not a bird

            # get bbox of bird in original image
            bbox = obj.bbox.scale(det_scale[0], det_scale[1])
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            # classify one detected bird
            img_bird = img_rgb[y0:y1, x0:x1, :]  # get image of bird
            if img_bird.size == 0:
                continue  # invalid detection
            img_bird_scaled = cv2.resize(img_bird, inference_pigeon_size)  # scale for inference
            run_inference(interpreter_pigeon, img_bird_scaled.tobytes())  # classification of bird
            bird_classes = classify.get_classes(interpreter_pigeon, args.top_k, args.threshold)  # get classes
            if len(bird_classes) == 0:
                continue  # not classified as bird

            # get class of bird with the highest score
            bird_top_1 = labels_bird.get(bird_classes[0].id, bird_classes[0].id)
            if 'Dove' or 'Pigeon' in bird_top_1:
                pigeon_detected()

            # draw bbox in original frame with bird class of the highest score
            if file_writer is not None:
                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                frame = cv2.putText(frame, bird_top_1, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                for c in bird_classes:  # log other classes to console (if arg.top_k > 1)
                    bird_label = labels_bird.get(c.id, c.id)
                    log.debug('%06d %s: %.5f' % (frame_idx, bird_label, c.score))

        t2 = time.perf_counter()

        # write frame to output
        if file_writer is not None:
            draw_debug_info(frame, frame_idx, t2 - t1)
            file_writer.write(frame)
        frame_idx += 1

    cap.release()


if __name__ == '__main__':
    main()
