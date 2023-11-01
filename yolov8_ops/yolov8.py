from yolo.utils.tools import draw_boxes_v8 
from skimage.transform import resize
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
from yolo.utils.tools import preprocess_image
import numpy as np
import imageio, tempfile
import shutil
import pandas as pd 
import cv2

def yolov8(st, df, shape, show, response, resume, return_sequence, colors, **kwargs):
    score_threshold = kwargs['score_threshold']
    yolo_model_v8   = YOLO('./yolov8/yolov8n.pt')
    frame           = kwargs['image_file'][0][0].copy()
    detections      = yolo_model_v8(frame)[0]
    boxes           = []
    box_classes     = []
    scores          = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >=  score_threshold:
            box_classes.append(int(class_id))
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    if scores:
        scores          = tf.constant(scores, dtype=tf.float32)
        box_classes     = tf.constant(box_classes, dtype=tf.int32)
        boxes           = tf.constant(boxes, dtype=tf.float32)
        class_names     = kwargs['Class_names']
        use_classes     = kwargs['class_names']

        image_predicted = draw_boxes_v8(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, 
                                        with_score=response, colors=colors,
                                        class_names=class_names, use_classes=use_classes, df=df, width=4)

        if len(shape) > 2 : shape = shape[:2]
        else: pass 
        
        image_predicted = resize(image_predicted, output_shape=shape)
    else:
        image_predicted = resize( np.array( kwargs['image_file'][0][0]), output_shape=shape)

    if return_sequence is False:
        resume(st=st, df=df, show=show, img = kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})
    else: return image_predicted

def yolov8_track(st, df, shape, show, response, resume, return_sequence, colors, tracker = None, track_num = [], **kwargs):
    score_threshold = kwargs['score_threshold']

    yolo_model_v8   = YOLO('./yolov8/yolov8n.pt')
    frame           = kwargs['image_file'][0][0].copy()
    detections      = yolo_model_v8.track(frame, conf=score_threshold, persist=True, tracker=tracker)[0]
    boxes           = []
    box_classes     = []
    scores          = []
    ids             = []
    
    for detection in detections.boxes.data.tolist():
        try:
            x1, y1, x2, y2, id_tracker, score, class_id = detection
        except ValueError: 
            x1, y1, x2, y2, score, class_id = detection
            id_tracker = -1

        if score >=  score_threshold:
            if int(id_tracker) in track_num:
                box_classes.append(int(class_id))
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                ids.append(int(id_tracker))

    if scores:
        scores          = tf.constant(scores, dtype=tf.float32)
        box_classes     = tf.constant(box_classes, dtype=tf.int32)
        boxes           = tf.constant(boxes, dtype=tf.float32)
        ids             = tf.constant(ids, dtype=tf.int32)
        class_names     = kwargs['Class_names']
        use_classes     = kwargs['class_names']

        image_predicted = draw_boxes_v8(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, 
                                        with_score=response, colors=colors, class_names=class_names, 
                                        use_classes=use_classes, df=df, width=4, ids=ids)

        image_predicted = resize(image_predicted, output_shape=shape)
    else:
        image_predicted = resize( np.array( kwargs['image_file'][0][0]), output_shape=shape)

    if return_sequence is False:
        resume(st=st, df=df, show=show, img = kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})
        
    else: return image_predicted

def yolovo_video(st, video, df, details, show, resume, response,  run, colors, **items):

    #storage             = []
    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    if run: 
        # progress bar 
        progress_text   = "Operation in progress. Please wait."
        my_bar          = st.progress(0, text=progress_text)

        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in enumerate(video):
                if i in range(start, end, step):
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame,frame_data)]
                    
                    image_predicted = yolov8(st=st, df=df, shape=shape, show=show, response=response,
                                            resume=None, return_sequence=True, colors=colors, **items)
                    
                    #image_predicted = np.uint8(255 * image_predicted )
                    # Écrire le tableau NumPy dans une vidéo avec imageio
                    #with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
                    image_predicted = image_predicted.astype('float32')
                    writer.append_data(image_predicted)

                    if i <= 100:
                        my_bar.progress(i, text=progress_text)
                    else: pass
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
        
def yolov8_video_track(st, video, df, details, show, resume, response,  run, colors, tracker, track_num, **items):
    #storage             = []
    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    #s = st.empty()
    if run: 
        # progress bar 
        progress_text   = "Operation in progress. Please wait."
        my_bar          = st.progress(0, text=progress_text)
        idd = -1
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in enumerate(video):
                if i in range(start, end, step):
                    idd += 1
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame,frame_data)]
                    
                    image_predicted = yolov8_track(st=st, df=df, shape=shape, show=show, response=response,
                                            resume=None, return_sequence=True, colors=colors, tracker=tracker, track_num=track_num, **items)
                    #import streamlit as st 
                    
                    #s.image(image_predicted)
                    #image_predicted = image_predicted.astype('float32')
                    #if idd > 50 : break
                    image_predicted = image_predicted.astype('float32')
                    writer.append_data(image_predicted)

                    #storage.append(image_predicted)

                    if i <= 100:
                        my_bar.progress(i, text=progress_text)
                    else: pass
                else: pass

        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        my_bar.empty()

        if video_data:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass    
    else: pass