import tensorflow as tf
import numpy as np

def yolo_filter_boxes(
        boxes : np.ndarray, 
        box_confidence : np.ndarray,  
        box_class_prob : np.ndarray,
        threshold : float = 0.6
        ) -> tuple :
    
    """
    * ----------------------------------------------------------------------------- 

    >>> AUTOR : < Iréné Amiehe-Essomba > 
    >>> Copyright (c) 2023

    * -----------------------------------------------------------------------------

    boxes.shape          = (1, n, m, l, 4) where 4 refers to (bx, by, bh, bw) 
                                        the coordinates of the bonding boxes 
    box_confidence.shape = (1, n, m, l, 1) where 1 is pc
    box_class_prob.shape = (1, n, m, l, classes)

    l = is the nombers of bonding boxes 
    (n, m) resprents the dimension of the final image after (ConvNet + MaxPool) operation
    usually time (m == n) 

    note that to create filter box as describe below we saparate y matrix in three matrices

    y = [pc, bx, by, bh, bw, c1, c2, c3, c4, ..............., c78, c79, c80] in this case we 
        suppose that we have 80 class probalities

    y = [[box_confidence], [boxes], [box_class_prob]]

    >>> box_confidence contains all pc values for all class probabilities
    >>> boxes contains all the [bx, by, bh, bw] values for all the bonding boxes
    >>> box_class_prob contains all the class probabilities 

    box_score = PC * box_class_prob
          = box_confidence * box_class_prob

    then we can find the index and the score using np.argmax() and tf.reduce_max()
    """
    
    # computing the score 
    # box_score.shape = (1, n, m, l, classes)
    box_scores = box_confidence * box_class_prob 

    # find index with the highest scores belong to the classes 
    # box_classes.shape = (1, n, m, l, 1)
    box_classes = np.argmax(box_scores, axis=-1)

    # find the score of the corresponding index 
    # box_class_scores.shape = (1, n, m, l)
    box_class_scores = tf.reduce_max(box_scores, axis=-1, keepdims=False)
   
    # create a mask using the threshold to revome the boxes with a score lower than the threshold
    # The mask should have the same dimension as box_class_scores, and be True for the boxes you 
    # want to keep (with probability >= threshold) 

    mask = box_class_scores >= threshold

    # using this mask to return the final result 
    # the mask should be applied on box_class_scores, boxes and box_classes
    # N is the number of bondings boxese
    # score (N)
    scores      = tf.boolean_mask(tensor=box_class_scores, mask=mask, axis=None)
    # score (N, 4)
    boxes       = tf.boolean_mask(tensor=boxes, mask=mask, axis=None)
    # score (N,)
    classes     = tf.boolean_mask(tensor=box_classes, mask=mask, axis=None)
   
    return scores, boxes, classes

