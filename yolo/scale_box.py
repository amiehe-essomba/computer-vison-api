import tensorflow as tf 
import numpy as np 

def scale_boxes(
        boxes : np.ndarray, 
        image_shape : tuple 
        ):
    """ Scales the predicted boxes in order to be drawable on the image"""

    height          = float(image_shape[0])
    width           = float(image_shape[1])
    image_dims      = tf.keras.backend.stack([height, width, height, width])
    image_dims      = tf.keras.backend.reshape(image_dims, [1, 4])
    boxes           = boxes * image_dims
    
    return boxes