import tempfile
import imageio
from PIL import Image
from yolo.utils.tools import preprocess_image
import tensorflow as tf 
from skimage.transform import resize
from yolo.utils.tools import draw_boxes_v8, draw_boxes_v8_seg
import numpy as np 
from collections import defaultdict
import cv2
from ultralytics import YOLO
import shutil
from yolo.predictions import prediction
from yolov8_ops.yolov8_seg import draw_mask 
import logging

class Wrapper:
    def __init__(self, model_name : str = '') -> None:
        self.model_name = model_name

    def models(self, is_seg:bool=False):
        keys = ["yolov8n-seg.pt", 'yolov8n.pt', 'my_model']
        if self.model_name in keys:
            if self.model_name in keys[:2]:
                if is_seg is False:
                    if self.model_name == keys[1]:
                        return YOLO(f'./yolo_model/{self.model_name}')
                    else:
                        print(f"is_seg is False but model_name != 'yolov8n.pt'")
                        return None
                else:
                    if keys[0] == self.model_name:
                        return YOLO(f'./yolo_model/{self.model_name}')
                    else:
                        print(f"is_seg is True but model_name != 'yolov8n-seg.pt'")
            else:
                yolo_model_path = './yolo_model/'
                tf.get_logger().setLevel(logging.ERROR)
                return tf.keras.models.load_model(yolo_model_path, compile=False)
        else:
            print(f"model name not in {keys}")
            return None 
    
    def tracker_check(self):
        self.tracker_val = ("bytetrack.yaml", "botsort.yaml")
        if self.tracker is not None:
            if self.tracker in self.tracker_val:
                return self.tracker 
            else:
                print(f"tracker not in the list : {self.tracker_val}")
                return True 
        else: 
            return self.tracker
        
class YOLO_MODEL:
    def __init__(self) -> None:
        pass

    def yolov8_demo(self, df, shape, response,  colors, model, **items):
        score_threshold = items['score_threshold']
        frame           = items['image_file'][0][0].copy()
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
            class_names     = items['Class_names']
            use_classes     = items['class_names']

            image_predicted = draw_boxes_v8(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, 
                                            with_score=response, colors=colors,
                                            class_names=class_names, use_classes=use_classes, df=df, width=4)

            if len(shape) > 2 : shape = shape[:2]
            else: pass 
            
            image_predicted = resize(image_predicted, output_shape=shape)
        else:
            image_predicted = resize( np.array( items['image_file'][0][0]), output_shape=shape)

        return image_predicted
    
    def yolov8_seg_demo(self, df, shape, response, colors, model, **items):
        frame           = items['image_file'][0][0].copy()
        detections      = model.predict(frame)[0]
        score_threshold = items['score_threshold']
        alpha           = items['alpha']
        mode            = items['mode']
        only_mask       = items['only_mask']
        with_names      = items['with_names']
    
        boxes           = []
        box_classes     = []
        scores          = []
    
        masks           = detections.masks.data.numpy()
        class_id        = detections.boxes.data.numpy()[:, -1].astype("int32")
        frame = draw_mask(frame, masks=masks, Colors=colors, class_names=items['Class_names'], alpha=100, class_id = class_id, mode=mode)

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
            class_names     = items['Class_names']
            use_classes     = items['class_names']

            if with_names is True or only_mask is False:
                image_predicted = draw_boxes_v8_seg(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                    class_names=class_names, use_classes=use_classes, df=df, colors=colors, alpha=alpha, only_mask=only_mask, with_names=with_names)
            else:
                image_predicted = np.array(frame)
            if len(shape) > 2 : shape = shape[:2]
            else: pass 

            image_predicted = resize(image_predicted, output_shape=shape)
        else:  
            image_predicted = items['image_file'][0][0]

        return image_predicted

    def yolov8_track_demo(self, df, shape,  response, colors, tracker = None, 
                                track_history = None, model=None, **items):
        score_threshold = items['score_threshold']
        frame           = items['image_file'][0][0].copy()
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
            class_names     = items['Class_names']
            use_classes     = items['class_names']

            image_predicted = draw_boxes_v8(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, 
                                            with_score=response, colors=colors, class_names=class_names, 
                                            use_classes=use_classes, df=df, width=4, ids=ids)

            image_predicted = resize(image_predicted, output_shape=shape)
        else:
            image_predicted = resize( np.array( items['image_file'][0][0]), output_shape=shape)

        return image_predicted, points

    def yolovo_video_demo(self, video, df,  details, response, colors, model, save, **items):
        
        frame_count         = 0
        fps                 = items["fps"]
        (start, end, step)  = details
        temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_data          = None 
        i                   = -1

        with imageio.get_writer(temp_video_file.name,  fps=fps, format='FFMPEG', mode='I') as writer:
            while (video.isOpened()):
                re, frame = video.read()
                if re:
                    i += 1
                    
                    if i in range(start, end, step):
                        frame  = Image.fromarray(frame, mode='RGB')
                        frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                        frame_count                += 1
                        items['image_file']         = [(frame,frame_data)]
                        
                        image_predicted = self.yolov8_demo(df=df, shape=shape, response=response, 
                                                                        colors=colors, model=model, **items)
                        image_predicted = image_predicted.astype('float32')
                        writer.append_data(image_predicted)
                
                    else: pass

                    if i == end: break
                else: break
        
        print("\n\n>>>> banary lecture in progress ...")
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
        print(">>>> process complete .....\n")
        print(">>>> banary writing in progress ...")
        with open(save, 'wb') as out_file:
            # Lire le contenu du fichier temporaire
           out_file.write(video_data)

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        video.release()
        print(">>>> process complete .....\n")
        print(f"\n\n>>>> Video saved to: {save}")
        return video_data
    
    def yolovo_video_seg_demo(self, video, df,  details, response, colors, model, save, **items):
        
        frame_count         = 0
        fps                 = items["fps"]
        (start, end, step)  = details
        temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_data          = None 
        i                   = -1

        with imageio.get_writer(temp_video_file.name,  fps=fps, format='FFMPEG', mode='I') as writer:
            while (video.isOpened()):
                re, frame = video.read()
                if re:
                    i += 1
                    
                    if i in range(start, end, step):
                        frame  = Image.fromarray(frame, mode='RGB')
                        frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                        frame_count                += 1
                        items['image_file']         = [(frame,frame_data)]
                        
                        image_predicted = self.yolov8_seg_demo(df=df, shape=shape, response=response,
                                        colors=colors,  model=model, **items)
                        image_predicted = image_predicted.astype('float32')
                        writer.append_data(image_predicted)
                
                    else: pass

                    if i == end: break
                else: break
        
        print("\n\n>>>> banary lecture in progress ...")
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
        print(">>>> process complete .....\n")
        print(">>>> banary writing in progress ...")
        with open(save, 'wb') as out_file:
            # Lire le contenu du fichier temporaire
           out_file.write(video_data)

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        video.release()
        print(">>>> process complete .....\n")
        print(f"\n\n>>>> Video saved to: {save}")
        return video_data

    def yolovo_video_track_demo(self, video, df,  details, response, colors, model, tracker, save, **items):
        
        frame_count         = 0
        fps                 = items["fps"]
        (start, end, step)  = details
        temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_data          = None 
        i                   = -1
        color_track         = (0, 255, 0)
        track_history       = defaultdict(lambda: []) 
        
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
                        
                        image_predicted, points = self.yolov8_track_demo(df=df, shape=shape, response=response,
                                        colors=colors, tracker=tracker, track_history=track_history, model=model, **items)
                      
                        image_predicted = (np.array(image_predicted) * 255).astype(np.int32)

                        if points.all():
                            cv2.polylines(image_predicted, [points], isClosed=False, color=color_track, thickness=10)

                        image_predicted = np.uint8(image_predicted)
                        writer.append_data(image_predicted)
                 
                    else: pass

                    if i == end: break
                else: break
        
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        print("\n\n>>>> banary lecture in progress ...")
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
        print(">>>> process complete .....\n")
        print(">>>> banary writing in progress ...")
        with open(save, 'wb') as out_file:
            # Lire le contenu du fichier temporaire
           out_file.write(video_data)

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        video.release()
        print(">>>> process complete .....\n")
        print(f"\n\n>>>> Video saved to: {save}")

        return video_data
    
    def my_model(self, video, df,  details, response, colors, model, save, **items):
        frame_count         = 0
        fps                 = items["fps"]
        (start, end, step)  = details
        temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_data          = None 
        i                   = -1

        with imageio.get_writer(temp_video_file.name,  fps=fps, format='FFMPEG', mode='I') as writer:
            while (video.isOpened()):
                re, frame = video.read()
                if re:
                    i += 1
                    
                    if i in range(start, end, step):
                        frame  = Image.fromarray(frame, mode='RGB')
                        frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True) 
                        frame_count                += 1
                        items['image_file']         = [(frame,frame_data)]
                        
                        image_predicted = prediction(yolo_model=model, use_classes=items['class_names'],
                                image_file=[(frame, frame_data)], anchors=items['anchors'], 
                                class_names=items['Class_names'], img_size=(608, 608),
                                max_boxes=items['max_boxes'], score_threshold=items['score_threshold'], 
                                iou_threshold=items['iou_threshold'], data_dict=df,shape=shape[:-1], 
                                file_type='video', with_score = response, colors=colors
                                )
                        image_predicted = image_predicted.astype('float32')
                        writer.append_data(image_predicted)
                
                    else: pass

                    if i == end: break
                else: break
        
        print("\n\n>>>> banary lecture in progress ...")
        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
        print(">>>> process complete .....\n")
        print(">>>> banary writing in progress ...")
        with open(save, 'wb') as out_file:
            # Lire le contenu du fichier temporaire
           out_file.write(video_data)

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        video.release()
        print(">>>> process complete .....\n")
        print(f"\n\n>>>> Video saved to: {save}")
        return video_data
  
    
    