import tensorflow as tf
import numpy as np 

def yolo_non_max_suppression(
        scores          : np.ndarray, 
        boxes           : np.ndarray, 
        classes         : np.ndarray, 
        max_boxes       : int = 10, 
        iou_threshold   : float = 0.5
        ):
    """
    * ----------------------------------------------------------------------------- 

    >>> AUTOR : < Iréné Amiehe-Essomba > 
    >>> Copyright (c) 2023

    * -----------------------------------------------------------------------------
    """

    # tensor to be used in tf.image.non_max_suppression()
    max_boxes_tensor = tf.Variable(max_boxes, dtype=tf.int32) 

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices     = tf.image.non_max_suppression( boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold, name=None)
    
    # score
    scores      = tf.gather(scores,indices = nms_indices)
    # boxes
    boxes       = tf.gather(boxes,indices = nms_indices)
    # classes
    classes     = tf.gather(classes, indices = nms_indices)
 
    return scores, boxes, classes