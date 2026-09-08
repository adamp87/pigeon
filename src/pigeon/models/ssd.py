from __future__ import annotations

from pigeon.models.yolo import BBox, Object


def decode_ssd_predictions(
    interpreter,
    score_threshold: float = 0.5,
    det_scale: tuple = (1.0, 1.0),
) -> list[Object]:
    """
    Decodes standard SSD MobileDet TFLite output tensors:
    - Tensor 0: [1, N, 4] bounding boxes [ymin, xmin, ymax, xmax] (normalized 0..1)
    - Tensor 1: [1, N] class ids
    - Tensor 2: [1, N] scores
    - Tensor 3: [1] count
    """
    output_details = interpreter.get_output_details()

    # PyCoral or generic TFLite SSD output layout
    boxes_tensor = interpreter.get_tensor(output_details[0]["index"])[0]
    classes_tensor = interpreter.get_tensor(output_details[1]["index"])[0]
    scores_tensor = interpreter.get_tensor(output_details[2]["index"])[0]
    count = int(interpreter.get_tensor(output_details[3]["index"])[0])

    objects = []
    for i in range(min(count, len(scores_tensor))):
        score = float(scores_tensor[i])
        if score < score_threshold:
            continue
        class_id = int(classes_tensor[i])
        ymin, xmin, ymax, xmax = boxes_tensor[i]

        objects.append(
            Object(
                id=class_id,
                score=score,
                bbox=BBox(
                    xmin=float(xmin * det_scale[0]),
                    ymin=float(ymin * det_scale[1]),
                    xmax=float(xmax * det_scale[0]),
                    ymax=float(ymax * det_scale[1]),
                ),
            )
        )
    return objects
