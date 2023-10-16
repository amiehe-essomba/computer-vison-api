from yolo.corner import yolo_boxes_to_corners
from yolo.filter_boxes import yolo_filter_boxes
from yolo.scale_box import scale_boxes
from yolo.non_max import yolo_non_max_suppression 

def yolo_eval(
        yolo_outputs    : list , 
        image_shape     : tuple = (720, 1280), 
        max_boxes       : int   = 10, 
        score_threshold : float = .6, 
        iou_threshold   : float = .5
        ):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates (bx, by, bh, bw)
    classes -- tensor of shape (None,), predicted class for each box (c1, c2, c3, ....., cN), where N is classes probabilities
    """

    # outputs of the YOLO model 
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Converting boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Using one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold 
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = score_threshold)
    
    # Scaling boxes back to original image shape.
    boxes = scale_boxes(boxes=boxes, image_shape=image_shape)
    
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold 
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, 
                                                      iou_threshold = iou_threshold)
  
    return scores, boxes, classes