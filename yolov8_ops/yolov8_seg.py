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

def yolov8_seg(st, df, shape, show, response, resume, return_sequence, colors, alpha, mode, seg=False, **kwargs):
    import streamlit as st
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
    frame = draw_mask(frame, masks=masks, Colors=colors, class_names=kwargs['Class_names'], alpha=100, class_id = class_id)
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

        image_predicted = draw_boxes_v8_seg(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                                        class_names=class_names, use_classes=use_classes, df=df, colors=colors, alpha=alpha, mode=mode)
        
        if len(shape) > 2 : shape = shape[:2]
        else: pass 

        image_predicted = resize(image_predicted, output_shape=shape)
    else:  
        image_predicted = kwargs['image_file'][0][0]

    if return_sequence is False:
        resume(st=st, df=df, show=show, img = kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})
    else: return image_predicted

def yolovo_video_seg(st, video, df, details, show, resume, response,  run, colors, alpha, mode, **items):
    frame_count         = 0
    fps                 = video.get_meta_data()['fps']
    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    if run: 
        # progress bar 
        progress_text   = "Operation in progress. Please wait."
        my_bar          = st.progress(0, text=progress_text)
        #s = st.empty()
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for i, frame in enumerate(video):
                if i in range(start, end, step):
                    frame  = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                    frame_count                += 1
                    items['image_file']         = [(frame,frame_data)]
                    
                    image_predicted= yolov8_seg(st=st, df=df, shape=shape, show=show, response=response,
                                            resume=None, return_sequence=True, colors=colors, alpha=alpha,mode=mode, seg=True, **items)
                    #s.image(image_predicted)
                    #image_predicted = image_predicted.astype('float32')
                    writer.append_data(image_predicted)
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

def draw_mask(image, masks, Colors, class_names, alpha=80, class_id=None):
    from PIL import ImageDraw
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

    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)
    
    return result











