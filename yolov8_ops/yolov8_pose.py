from yolo.utils.tools import  draw_boxes_v8
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

def yolov8_pose(
        st              :st,  
        df              :pd.DataFrame, 
        colors          :dict,  
        radus           :int, 
        line_width      :int, 
        shape           :tuple, 
        resume          :pd.DataFrame, 
        return_sequence :bool = False, 
        show            :bool = True, 
        od              :bool = False,
        response        :bool = False,
        model           :any  = None,
        font            : str = 'calibril.ttf',
        **kwargs
        ) -> None | np.ndarray:
    
    frame           = kwargs['image_file'][0][0].copy()
    detections      = model.predict(frame)[0]
    
    
    if od:
        obj = object_detection(frame=frame, detections=detections, 
                            df=df, colors=colors, response=response, width=line_width, font=font, **kwargs)
    try:
        keypoints = detections.keypoints.data.numpy()
        ndim = keypoints.shape[0]
        
        for i in range(ndim):
            frame = connections(image=frame, kpts=keypoints[i].reshape((17, 3)), colors=colors, point_radius=radus, width=line_width)

        image_predicted = resize(np.array(frame), output_shape=shape)
    except Exception: 
        image_predicted = kwargs['image_file'][0][0]

    if return_sequence is False:
        resume(st=st, df=df, show=show, img = kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})
    else: return image_predicted

def object_detection(frame, detections, df, colors, response, width, font, **kwargs):
    box_classes, boxes, scores = [[], [], []]
    score_threshold = kwargs['score_threshold']
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= score_threshold :
            box_classes.append(int(class_id))
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    if scores:
        scores          = tf.constant(scores, dtype=tf.float32)
        box_classes     = tf.constant(box_classes, dtype=tf.int32)
        boxes           = tf.constant(boxes, dtype=tf.float32)
        class_names     = kwargs['Class_names']
        use_classes     = kwargs['class_names']
        image_predicted = draw_boxes_v8(image=frame, boxes=boxes, box_classes=box_classes, scores=scores, with_score=response,
                class_names=class_names, use_classes=use_classes, df=df, colors=colors, width=width, return_sequence=True, f=font, pose=True)
    else:
        image_predicted = frame
    
    del scores
    del boxes
    del box_classes

    return image_predicted

def connections(image, kpts, shape=(640, 640), point_radius=5, kpt_line=True, colors = None, width=2):
    from PIL import Image, ImageDraw
    import math
    """
    Plot keypoints on the image.

    Args:
        kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                    for human pose. Default is True.

    Note: `kpt_line=True` currently only supports human pose plotting.
    """
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    #self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    #self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    
    #if self.pil:
    #    # Convert to numpy first
    #    self.im = np.asarray(self.im).copy()

    colors = list(colors.values())
    palette = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
    temp_images = []

    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3

    # `kpt_line=True` for now only supports human pose plotting
    kpt_line &= is_pose  

    for i, k in enumerate(kpts):
        ##############
        temp_image  = Image.new("RGBA", image.size, (0, 0, 0, 0))
        temp_draw   = ImageDraw.Draw(temp_image) 
        ###############
        color_k = colors[palette[i]] #[int(x) for x in kpt_color[i]] if is_pose else colors(i)
        x, y = k[0], k[1]
        if x % shape[1] != 0 and y % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5:
                    continue
            
            temp_draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color_k)
            temp_images.append(temp_image)
     
    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            ###############
            temp_image  = Image.new("RGBA", image.size, (0, 0, 0, 0))
            temp_draw   = ImageDraw.Draw(temp_image)
            ###############
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue

            temp_images.append(temp_image)
            temp_draw.line([pos1, pos2], fill=colors[palette[i]]+(70, ), width=width)

    result = image.convert("RGBA")

    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)

        
    return result.convert("RGB")

