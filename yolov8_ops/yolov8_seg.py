from yolo.utils.tools import  draw_boxes_v8_seg
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
import streamlit as st

def line(a, b):
    # y1 = -3.05 * x + 860
    # y2 = 1.75 * x - 831

    isin = False 

    if (a[1] + 3.05 * a[0]) >= 800:
        if (b[1] - 2.7 * b[0]) >= -831:
            isin = True 

    return isin

    
    c1 = ()
    a = (alpha[1] - c ) / alpha[0]

    return (a, c)

def yolov8_seg(st:st, df, shape, show, response, resume, return_sequence, colors, alpha, mode, only_mask, with_names, model, **kwargs):
    frame           = kwargs['image_file'][0][0].copy()
    detections      = model.predict(frame)[0]
    score_threshold = kwargs['score_threshold']
    
   
    boxes           = []
    box_classes     = []
    scores          = []
   
    masks           = detections.masks.data.numpy()
    #if seg is True:
    class_id        = detections.boxes.data.numpy()[:, -1].astype("int32")
    #else: pass 

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold:
            box_classes.append(int(class_id))
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    frame = draw_mask(frame, masks=masks, Colors=colors, class_names=kwargs['Class_names'],
                       alpha=100, class_id = class_id, mode=mode, boxes=None)
    
    if scores:
        scores          = tf.constant(scores, dtype=tf.float32)
        box_classes     = tf.constant(box_classes, dtype=tf.int32)
        boxes           = tf.constant(boxes, dtype=tf.float32)
        class_names     = kwargs['Class_names']
        use_classes     = kwargs['class_names']

        if with_names is True or only_mask is False:
            image_predicted = draw_boxes_v8_seg(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                class_names=class_names, use_classes=use_classes, df=df, colors=colors, alpha=alpha, only_mask=only_mask, with_names=with_names)
        else:
            image_predicted = np.array(frame)
        if len(shape) > 2 : shape = shape[:2]
        else: pass 

        image_predicted = resize(image_predicted, output_shape=shape)
    else:  
        image_predicted = kwargs['image_file'][0][0]

    if return_sequence is False:
        resume(st=st, df=df, show=show, img = kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})
    else: return image_predicted

def demo_seg(df, shape,  response,  colors, alpha, mode, only_mask, with_names, **kwargs):
    
    yolo_model_v8   = YOLO('./yolov8/yolov8n-seg.pt')
    frame           = kwargs['image_file'][0][0].copy()
    detections      = yolo_model_v8.predict(frame)[0]
    score_threshold = kwargs['score_threshold']
    
   
    boxes           = []
    box_classes     = []
    scores          = []
   
    masks           = detections.masks.data.numpy()
    #if seg is True:
    class_id        = detections.boxes.data.numpy()[:, -1].astype("int32")
    frame = draw_mask(frame, masks=masks, Colors=colors, class_names=kwargs['Class_names'], 
                        alpha=100, class_id = class_id, mode=mode)
    #else: pass 

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

        if with_names is True or only_mask is False:
            image_predicted = draw_boxes_v8_seg(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                class_names=class_names, use_classes=use_classes, df=df, colors=colors, alpha=alpha, only_mask=only_mask, with_names=with_names)
        else:
            image_predicted = np.array(frame)
        if len(shape) > 2 : shape = shape[:2]
        else: pass 

        image_predicted = resize(image_predicted, output_shape=shape)
    else:  
        image_predicted = kwargs['image_file'][0][0]

    return image_predicted

def yolovo_video_seg(st:st, video, df, details, show, resume, response,  
                     run, colors, alpha, mode, only_mask, with_names, model, **items):
    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    if run: 
        s = st.empty()
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in enumerate(video):
                if i in range(start, end, step):
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame,frame_data)]
                    
                    image_predicted= yolov8_seg(st=st, df=df, shape=shape, show=show, response=response,
                        resume=None, return_sequence=True, colors=colors, alpha=alpha,mode=mode, 
                                    only_mask=only_mask, with_names=with_names, model=model,**items)
                    
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
        s.write('complete')
        s.empty()

        if video_data:  resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass  
    else: pass

def yolovo_video_seg_youtube(st:st, video, df, details, show, resume, response,  
                     run, colors, alpha, mode, only_mask, with_names, model, **items):
    frame_count         = 0
    fps                 = items["fps"]
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    if run: 
        s = st.empty()
        i =-1
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
                        
                        image_predicted= yolov8_seg(st=st, df=df, shape=shape, show=show, response=response,
                            resume=None, return_sequence=True, colors=colors, alpha=alpha,mode=mode, 
                                        only_mask=only_mask, with_names=with_names, model=model,**items)
                        
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
        s.write('complete')
        s.empty()
        video.release()
        if video_data:  resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_data})
        else: pass  
    else: pass

def draw_mask(image, masks, Colors, class_names, alpha=80, class_id=None, mode: str="gray", boxes=None):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import streamlit as st 
    # Assurez-vous que les masques ont les mêmes dimensions que l'image de fond
    masks = [Image.fromarray((resize(masks[i, :, :], output_shape=image.size) > 0.5).astype(np.uint8)) for i in range(masks.shape[0])]
    
    #result = image.copy()
    #draw = ImageDraw.Draw(result)
    
    temp_images = []
    
    for i in range(len(masks)):
        temp_image  = Image.new("RGBA", image.size, (0, 0, 0, 0))
        temp_draw   = ImageDraw.Draw(temp_image) 
        mask = masks[i]
        color = Colors[class_names[class_id[i]]]
        draw_mask = True 

        if boxes:
            left, top, right, bottom = boxes[i]
            if top > 250 and bottom <= 600:
                a, b = [left, bottom], [right, bottom]
                draw_mask = line(a=a, b=b)
            else: draw_mask = False

        if draw_mask:
            for x in range(mask.width):
                for y in range(mask.height):
                    if mask.getpixel((x, y)) > 0:
                        # Récupérez la couleur du pixel existant
                        r, g, b = color
                        # Créez une nouvelle couleur avec la valeur alpha spécifiée
                        color_with_alpha = (r, g, b, alpha)
                        temp_draw.point((x, y), fill=color_with_alpha)
            temp_images.append(temp_image)
    
    result = image.convert("RGBA")

    if mode == "gray": result = ImageOps.grayscale(image.copy()).convert('RGBA') 
   
    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)
    
    return result











