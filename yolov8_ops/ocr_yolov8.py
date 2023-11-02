from yolo.utils.tools import draw_boxes_v8, draw_boxes_v8_seg
from skimage.transform import resize
from ultralytics import YOLO
import tensorflow as tf
import cv2
from ocr_modules.utils import read_license_plate
import numpy as np

from yolo.utils.tools import preprocess_image
import numpy as np
import imageio, tempfile
import shutil
import tensorflow as tf
from PIL import Image
import cv2

def ocr_yolov8(st, df, shape, show, response, resume, scaling, return_sequence, colors, **kwargs):
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
    detections      = yolo_model_v8.predict(frame)[0]

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

    
        _ = draw_boxes_v8(image=frame, boxes=boxes_plates, box_classes=box_classes_plates, scores=scores_plates, 
                            with_score=response, class_names=CLASSES, use_classes=CLASSES, colors=colors,
                            df=df, C=(255, 255, 0), return_sequence=True, width = 4)
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
                    class_names=class_names, use_classes=use_classes, df=df, C = None, return_sequence=False, width=4, colors=colors)
    else:
        image_predicted = kwargs['image_file'][0][0]
    
    image_predicted = resize(image_predicted, output_shape=shape)

    if return_sequence is False:
        resume(st=st, df=df, show=show, img=kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})
    else:  return image_predicted 
    
    del boxes
    del scores 
    del box_classes
    del class_names
    del use_classes

def ocr_yolovo_video(st, video, df, details, show, resume, scaling, response,  run, colors, **items):
    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    #s= st.empty()

    if run: 
        # progress bar 
        progress_text   = "Operation in progress. Please wait."
        my_bar          = st.progress(0, text=progress_text)

        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in enumerate(video):
                if i in range(start, end, step):
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True, factor=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame, frame_data)]
                    
                    image_predicted = ocr_yolov8(st=st, df=df, shape=shape, show=show, response=response, scaling=scaling,
                                            resume=None, return_sequence=True, colors=colors, **items)
                    
                    image_predicted = image_predicted.astype('float32')
                    writer.append_data(image_predicted)
                    #s.image(image_predicted)
                    '''
                    if i <= 100:
                        my_bar.progress(i, text=progress_text)
                    else: pass
                    '''
                else: pass
        
        
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        my_bar.empty()

        if video_data:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass   
        
    else: pass