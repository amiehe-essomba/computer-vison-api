from yolo.utils.tools import draw_boxes_v8 
from skimage.transform import resize
import tensorflow as tf
from PIL import Image
from yolo.utils.tools import preprocess_image
import numpy as np
import imageio, tempfile
import shutil 
import cv2
from collections import defaultdict
import streamlit as st 
from stqdm import stqdm 


def yolov8(st, df, shape, show, response, resume, return_sequence, colors, model, **kwargs):
    score_threshold = kwargs['score_threshold']
    frame           = kwargs['image_file'][0][0].copy()
    detections      = model.predict(frame)[0]
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

def yolov8_track(st, df, shape, show, response, resume, return_sequence, colors, tracker = None, 
                                track_history = None, model=None, **kwargs):
    score_threshold = kwargs['score_threshold']
    #yolo_model_v8   = YOLO('./yolov8/yolov8n.pt')
    frame           = kwargs['image_file'][0][0].copy()
    detections      = model.track(frame, conf=score_threshold, persist=True, tracker=tracker)[0]
    boxes           = []
    box_classes     = []
    scores          = []
    ids             = []
    points          = None 

    for detection in detections.boxes.data.tolist():
        try:
            x1, y1, x2, y2, id_tracker, score, class_id = detection
            track = track_history[id_tracker]
            track.append((float(x1), float(y1)))   
            if len(track) > 30:   
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        except ValueError: 
            x1, y1, x2, y2, score, class_id = detection
            id_tracker = -1

        if score >=  score_threshold:
            #if int(id_tracker) in track_num:
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

    else: return image_predicted, points

def yolovo_video(st, video, df, details, show, resume, response,  run, colors, model, **items):

    #storage             = []
    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    s = st.empty()

    if run: 
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in enumerate(video):
                if i in range(start, end, step):
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame,frame_data)]
                    
                    image_predicted = yolov8(st=st, df=df, shape=shape, show=show, response=response,
                                    resume=None, return_sequence=True, colors=colors, model=model, **items)
                    
                    image_predicted = image_predicted.astype('float32')
                    writer.append_data(image_predicted)
                    s.write('banary writing in progress ...')
                else: pass

                if i == end: break
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
            s.write('banary lecture in progress ...')
        shutil.rmtree(temp_video_file.name, ignore_errors=True)

        if video_data:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass    
    else: pass
        
def yolov8_video_track(st:st, video, df, details, show, resume, response,  run, colors, tracker, track_num, model, **items):

    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
   
    s = st.empty()

    if run: 
        idd             = -1
        track_history   = defaultdict(lambda: []) 
        color_track     = list(colors.values())[-1]

        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in stqdm(enumerate(video), backend=False, frontend=True):
                if i in range(start, end, step):
                    idd += 1
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame,frame_data)]
    
                    image_predicted, points = yolov8_track(st=st, df=df, shape=shape, show=show, response=response,
                                resume=None, return_sequence=True, colors=colors, tracker=tracker, 
                                track_history=track_history, model=model, **items)
                    #import streamlit as st 
                    image_predicted = (np.array(image_predicted) * 255).astype(np.int32)

                    if points.all():
                        cv2.polylines(image_predicted, [points], isClosed=False, color=color_track, thickness=10)

                    #image_predicted /= 255.
                    image_predicted = np.uint8(image_predicted)
                    writer.append_data(image_predicted)
                    s.write('banary writing in progress ...')
                else: pass
                
                if i == end: break
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
            s.write('banary lecture in progress ...')

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        #my_bar.empty()

        s.write('complete')
        s.empty()
        if video_data:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass  
        
    else: pass

def yolovo_video_youtube(st, video, df, details, show, resume, response,  run, colors, model, **items):

    frame_count         = 0
    fps                 = items["fps"]
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    s = st.empty()
    
    if run: 
        i   = -1
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            while (video.isOpened()):
                re, frame = video.read()
                if re:
                    i += 1
                    
                    if i in range(start, end, step):
                        frame  = Image.fromarray(frame, mode='RGB')
                        frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                        frame_count                += 1
                        items['image_file']         = [(frame,frame_data)]
                        
                        image_predicted = yolov8(st=st, df=df, shape=shape, show=show, response=response,
                                        resume=None, return_sequence=True, colors=colors, model=model, **items)
                        image_predicted = image_predicted.astype('float32')
                        writer.append_data(image_predicted)
                   
                        s.write('banary writing in progress ...')
                    else: pass

                    if i == end: break
                else: break
        
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
            s.write('banary lecture in progress ...')

        shutil.rmtree(temp_video_file.name, ignore_errors=True)

        s.write('complete')
        s.empty()
        
        video.release()
        if video_data:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass    
    else: pass

def yolovo_video_youtube_track(st:st, video, df, details, show, resume, response,  run, colors, tracker, track_num, model, **items):
    frame_count         = 0
    fps                 = items["fps"]
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    s = st.empty()
    
    if run: 
        i   = -1
        idd             = -1
        track_history   = defaultdict(lambda: []) 
        color_track     = list(colors.values())[-1]

        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            while (video.isOpened()):
                re, frame = video.read()
                if re:
                    i += 1
                    
                    if i in range(start, end, step):
                        frame  = Image.fromarray(frame, mode='RGB')
                        frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                        frame_count                += 1
                        items['image_file']         = [(frame,frame_data)]
                        
                        image_predicted, points = yolov8_track(st=st, df=df, shape=shape, show=show, response=response,
                                resume=None, return_sequence=True, colors=colors, tracker=tracker, 
                                track_history=track_history, model=model, **items)
                        #import streamlit as st 
                        image_predicted = (np.array(image_predicted) * 255).astype(np.int32)

                        if isinstance(points, np.ndarray):
                            if points.all():
                                cv2.polylines(image_predicted, [points], isClosed=False, color=color_track, thickness=4)

                        #image_predicted /= 255.
                        image_predicted = np.uint8(image_predicted)
                        writer.append_data(image_predicted)
                        s.write('banary writing in progress ...')
                    else: pass

                    if i == end: break
                else: break
        
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
            s.write('banary lecture in progress ...')

        shutil.rmtree(temp_video_file.name, ignore_errors=True)

        s.write('complete')
        s.empty()
        
        video.release()
        if video_data:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass    
    else: pass