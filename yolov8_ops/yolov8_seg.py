from yolo.utils.tools import  draw_boxes_v8_seg
from skimage.transform import resize
from ultralytics import YOLO
import tensorflow as tf

def yolov8_seg(st, df, shape, show, response, resume, colors, alpha, **kwargs):
    yolo_model_v8   = YOLO('./yolov8/yolov8n-seg.pt')
    frame           = kwargs['image_file'][0][0].copy()
    detections      = yolo_model_v8(frame)[0]
    score_threshold = kwargs['score_threshold']

    boxes           = []
    box_classes     = []
    scores          = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold:
            box_classes.append(int(class_id))
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    if scores:
        scores          = tf.constant(scores, dtype=tf.float32)
        box_classes     = tf.constant(box_classes, dtype=tf.int32)
        boxes           = tf.constant(boxes, dtype=tf.float32)
        class_names     = kwargs['Class_names']
        use_classes     = kwargs['class_names']

        image_predicted = draw_boxes_v8_seg(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                                        class_names=class_names, use_classes=use_classes, df=df, colors=colors, alpha=alpha)

        image_predicted = resize(image_predicted, output_shape=shape)
    else:  image_predicted = kwargs['image_file'][0][0]
    resume(st=st, df=df, show=show, img=kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})