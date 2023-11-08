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

def yolov8_pose(st:st,  colors, **kwargs):
    yolo_model_v8   = YOLO('./yolov8/yolov8n-pose.pt')
    frame           = kwargs['image_file'][0][0].copy()
    detections      = yolo_model_v8.predict(frame)[0]
    score_threshold = kwargs['score_threshold']
    
    boxes           = []
    box_classes     = []
    scores          = []

    #res_plotted = detections.plot()
    
    keypoints = detections.keypoints.data.numpy()
    ndim = keypoints.shape[0]
    
    for i in range(ndim):
        frame = kpts(image=frame, kpts=keypoints[i].reshape((17, 3)), colors=colors)
    st.image(frame)

def connect(image, keypoints : np.ndarray, width : int = 4, colors : list  = []):
    from PIL import Image, ImageDraw
    import math


    keypoints_shape = keypoints.shape 
    #keypoints   = [(x[0], x[1])  for j in range(keypoints_shape[0]) for x in keypoints[j]]
    temp_images = []

    # Calcul de la distance minimale pour connecter les keypoints
    min_distance = 20
    connections = []

    colors = list(colors.values())
    point_radius = 2
    for k in range(keypoints_shape[0]):
        keypoint   = [(x[0], x[1])  for x in keypoints[k]]
        # Dessiner des points
        for i, point in enumerate(keypoint):
            temp_image  = Image.new("RGBA", image.size, (0, 0, 0, 0))
            temp_draw   = ImageDraw.Draw(temp_image) 
            x, y = point
            temp_draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=colors[k])

            try:
                start_point = (x, y)
                end_point = keypoint[i+1]
                distance = math.sqrt((x - end_point[0]) ** 2 + (y - end_point[1]) ** 2)
                if distance < 100:
                    temp_draw.line([start_point, end_point], fill=colors[k], width=2)
            except Exception: pass 
            temp_images.append(temp_image)
        
        result = image.convert("RGBA")

    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)
    
    return result
  
def kpts(image, kpts, shape=(640, 640), point_radius=5, kpt_line=True, colors = None):
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
            temp_draw.line([pos1, pos2], fill=colors[palette[i]], width=2)

    result = image.convert("RGBA")

    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)

        
    return result

