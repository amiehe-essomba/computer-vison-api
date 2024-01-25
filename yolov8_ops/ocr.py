from yolo.utils.tools import draw_boxes_v8 
from skimage.transform import resize
from ultralytics import YOLO
import tensorflow as tf
import cv2
from ocr_modules.utils import read_license_plate
import numpy as np
import matplotlib.pyplot as plt

def ocr(st, df, shape, show, response, resume, scaling, colors, model, font='./font/FiraMono-Medium.otf', **kwargs):
    frame           = kwargs['image_file'][0][0].copy()
    score_threshold = kwargs['score_threshold']
    detections      = model(frame)[0]

    boxes_plates           = []
    box_classes_plates     = []
    scores_plates          = []
    CLASSES                = []
    imgs                   = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold :
            box_classes_plates.append(int(class_id))
            boxes_plates.append([x1, y1, x2, y2])
            scores_plates.append(score)


            license_plate_drop                      = np.array(frame)[int(y1) : int(y2), int(x1) : int(x2), :]
            license_plate_drop_gray                 = cv2.cvtColor(license_plate_drop, cv2.COLOR_BGR2GRAY)
            s, license_plate_drop_threshold         = cv2.threshold(license_plate_drop_gray, 60, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_score = read_license_plate( license_plate_drop_threshold  )
            
            if license_plate_text: 
                CLASSES.append(license_plate_text)
                imgs.append([license_plate_drop, license_plate_drop_threshold ])
    if CLASSES:
        boxes_plates, frame    = scaling(frame, boxes=boxes_plates, S=frame.size)
        scores_plates          = tf.constant(scores_plates, dtype=tf.float32)
        box_classes_plates     = tf.constant(box_classes_plates, dtype=tf.int32)
        boxes_plates           = tf.constant(boxes_plates, dtype=tf.float32)

        image_predicted  = draw_boxes_v8(image=frame, boxes=boxes_plates, box_classes=box_classes_plates, scores=scores_plates, 
                            with_score=response, class_names=CLASSES, use_classes=CLASSES, colors=colors,
                            df=df, C=(255, 255, 0), return_sequence=False, width = 1, f=font)
        
        image_predicted = resize(image_predicted, output_shape=shape)
    else:
        image_predicted = resize( np.array( kwargs['image_file'][0][0]), output_shape=shape)
    
    del imgs 

    resume(st=st, df=df, show=show, img=kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})

def ocr_demo(df, shape, response, scaling, colors, **kwargs):
    model           = YOLO('./yolov8/license_plate_detector.pt')
    frame           = kwargs['image_file'][0][0].copy()
    score_threshold = kwargs['score_threshold']
    detections      = model(frame)[0]

    boxes_plates           = []
    box_classes_plates     = []
    scores_plates          = []
    CLASSES                = []
    imgs                   = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold :
            box_classes_plates.append(int(class_id))
            boxes_plates.append([x1, y1, x2, y2])
            scores_plates.append(score)


            license_plate_drop                      = np.array(frame)[int(y1) : int(y2), int(x1) : int(x2), :]
            license_plate_drop_gray                 = cv2.cvtColor(license_plate_drop, cv2.COLOR_BGR2GRAY)
            s, license_plate_drop_threshold         = cv2.threshold(license_plate_drop_gray, 90, 245, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_score = read_license_plate(license_plate_drop_threshold)
            
            if license_plate_text: 
                CLASSES.append(license_plate_text)
                imgs.append([license_plate_drop, license_plate_drop_threshold   ])
    
    if CLASSES:
        boxes_plates, frame    = scaling(frame, boxes=boxes_plates, S=frame.size)
        scores_plates          = tf.constant(scores_plates, dtype=tf.float32)
        box_classes_plates     = tf.constant(box_classes_plates, dtype=tf.int32)
        boxes_plates           = tf.constant(boxes_plates, dtype=tf.float32)

        image_predicted  = draw_boxes_v8(image=frame, boxes=boxes_plates, box_classes=box_classes_plates, scores=scores_plates, 
                            with_score=response, class_names=CLASSES, use_classes=CLASSES, colors=colors,
                            df=df, C=(255, 255, 0), return_sequence=False, width = 4, imgs=imgs, ocr=True)
        image_predicted = resize(image_predicted, output_shape=shape)
    else:
        image_predicted = resize( np.array( kwargs['image_file'][0][0]), output_shape=shape)
    
    del imgs 

    return image_predicted

     