from yolo.utils.tools import draw_boxes_v8, draw_boxes_v8_seg
from skimage.transform import resize
from ultralytics import YOLO
import tensorflow as tf
import cv2
from ocr_modules.utils import read_license_plate
import numpy as np

def ocr_yolov8(st, df, shape, show, response, resume, scaling, **kwargs):
    yolo_model_v8   = YOLO('./yolov8/yolov8n.pt')
    yolo_model_ocr  = YOLO('./yolov8/license_plate_detector.pt')
    frame           = kwargs['image_file'][0][0].copy()
    score_threshold = kwargs['score_threshold']
    detections      = yolo_model_ocr(frame)[0]

    boxes_plates           = []
    box_classes_plates     = []
    scores_plates          = []
    CLASSES                = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold :
            box_classes_plates.append(int(class_id))
            boxes_plates.append([x1, y1, x2, y2])
            scores_plates.append(score)

            license_plate_drop                      = np.array(frame)[int(y1) : int(y2), int(x1) : int(x2), :]
            license_plate_drop_gray                 = cv2.cvtColor(license_plate_drop, cv2.COLOR_BGR2GRAY)
            s, license_plate_drop_threshold         = cv2.threshold(license_plate_drop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_score = read_license_plate(license_plate_drop_gray)
            
            if license_plate_text: 
                CLASSES.append(license_plate_text)
    
    boxes           = []
    box_classes     = []
    scores          = []
    detections      = yolo_model_v8(frame)[0]

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold:
            box_classes.append(int(class_id))
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    if CLASSES:
        boxes_plates, frame    = scaling(frame, boxes=boxes_plates, S=frame.size)
        scores_plates          = tf.constant(scores_plates, dtype=tf.float32)
        box_classes_plates     = tf.constant(box_classes_plates, dtype=tf.int32)
        boxes_plates           = tf.constant(boxes_plates, dtype=tf.float32)

        frame = draw_boxes_v8(image=frame, boxes=boxes_plates, box_classes=box_classes_plates, scores=scores_plates, 
                            with_score=response, class_names=CLASSES, use_classes=CLASSES, 
                            df=df, C=(255, 255, 0), return_sequence=True, width = 6)
    
    del boxes_plates
    del scores_plates 
    del box_classes_plates

    if boxes:
        boxes, frame    = scaling(frame, boxes=boxes,S=(shape[1], shape[0]))
        scores          = tf.constant(scores, dtype=tf.float32)
        box_classes     = tf.constant(box_classes, dtype=tf.int32)
        boxes           = tf.constant(boxes, dtype=tf.float32)
        class_names     = kwargs['Class_names']
        use_classes     = kwargs['Class_names']

        image_predicted = draw_boxes_v8(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                    class_names=class_names, use_classes=use_classes, df=df, C = None, return_sequence=False, width=4)
    else:
        image_predicted = kwargs['image_file'][0][0]
    
    image_predicted = resize(image_predicted, output_shape=shape)
    resume(st=st, df=df, show=show, img=kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})

    del boxes
    del scores 
    del box_classes
    del class_names
    del use_classes