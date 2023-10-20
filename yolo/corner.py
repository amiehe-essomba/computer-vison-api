import tensorflow as tf

def yolo_boxes_to_corners(box_xy, box_wh):
    """
    * ----------------------------------------------------------------------------- 

    >>> AUTOR : < Iréné Amiehe-Essomba > 
    >>> Copyright (c) 2023

    * -----------------------------------------------------------------------------
    
    Convert YOLO box predictions to bounding box corners.
    """
    box_mins  = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],   # y_min
        box_mins[..., 0:1],   # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]   # x_max
    ])