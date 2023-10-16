def code(name: str):

    ("IoU", "yolo filter boxes", "yolo-non-max suppression", 
                "yolo boxes to corners", "yolo evaluation", "yolo model")
    
    filter_box = '''
    import tensorflow as tf
    import numpy as np

    def yolo_filter_boxes(
            boxes : np.ndarray, 
            box_confidence : np.ndarray,  
            box_class_prob : np.ndarray,
            threshold : float = 0.6
            ) -> tuple :
        
        """
        * -----------------------------------------------------------------------------------------------------------

        >>> AUTOR : < Iréné Amiehe-Essomba > 
        >>> Copyright (c) 2023

        * -----------------------------------------------------------------------------------------------------------

        boxes.shape          = (n, m, l, 4) where 4 refers to (bx, by, bh, bw) the coordinates of the bonding boxes 
        box_confidence.shape = (n, m, l, 1)
        box_class_prob.shape = (n, m, l, classes)

        l = is the nombers of bonding boxes 
        (n, m) resprents the dimension of the final image after (ConvNet + MaxPool) operation

        note that to create filter box as describe below we saparate y matrix in three matrices

        y = [pc, bx, by, bh, bw, c1, c2, c3, c4, ..............., c78, c79, c80] in this case we suppose that we have 80 class probalities

        y = [[box_confidence], [boxes], [box_class_prob]]

        >>> box_confidence contains all pc values for all class probabilities
        >>> boxes contains all the [bx, by, bh, bw] values for all the bonding boxes
        >>> box_class_prob contains all the class probabilities 

        box_score = PC * box_class_prob
            = box_confidence * box_class_prob

        then we can find the index and the score using np.argmax() and tf.reduce_max()
        """

        # computing the score 
        box_scores = box_confidence * box_class_prob 

        # find index with the highest scores belong to the classes 
        box_classes = np.argmax(box_scores, axis=-1)

        # find the score of the corresponding index 
        box_class_scores = tf.reduce_max(box_scores, axis=-1, keepdims=False)

        # create a mask using the threshold to revome the boxes with a score lower than the threshold
        # The mask should have the same dimension as box_class_scores, and be True for the boxes you 
        # want to keep (with probability >= threshold) 
        
        mask = box_class_scores >= threshold

        # using this mask to return the final result 
        # the mask should be applied on box_class_scores, boxes and box_classes

        scores      = tf.boolean_mask(tensor=box_class_scores, mask=mask, axis=None)
        boxes       = tf.boolean_mask(tensor=boxes, mask=mask, axis=None)
        classes     = tf.boolean_mask(tensor=box_classes, mask=mask, axis=None)


        return scores, boxes, classes

    '''

    non_max    = '''
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
        '''
    
    iou        = '''

        def IoU(
            box_xy1 : tuple = (), 
            box_xy2 : tuple = ()
            ):

            """
            * ----------------------------------------------------------------------------- 

            >>> AUTOR : < Iréné Amiehe-Essomba > 
            >>> Copyright (c) 2023

            * -----------------------------------------------------------------------------

            (b1_x1, b1_y1)(min)
            +---------------------------------------------+
            |                                             |
            |                                             |
            |                                             |
            |      (b2_x1, b2_y1)(min)    box 2           |
            |        +----------------------------+-------+
            |        |/////// bonding box /////// |       |
            +--------+------------------------------------+ (b1_x2, b1_y2) (max) 
                    |                            |
                    |        box 1               |
                    |                            |
                    +----------------------------+ (b2_x2, b2_y2) (max)
            """
            
            # boxes coordinates 
            (b1_x1, b1_y1, b1_x2, b1_y2) = box_xy1
            (b2_x1, b2_y1, b2_x2, b2_y2) = box_xy2
            
            # bonding box coordinates
            xi1 = max(b1_x1, b1_x2)
            yi1 = max(b1_y1, b1_y2)

            xi2 = min(b2_x2, b2_x1)
            yi2 = min(b2_y1, b2_y2)

            inner_width             = (xi1 - xi2)
            inner_height            = (yi1 - yi2)

            # bonding box surface calculation
            bonding_box_surface     = max(inner_width, 0) * max(inner_height, 0)

            # surface area of each box 
            surface_box1            = b1_x2 * b1_y2
            surface_box2            = b2_x2 * b2_y2 

            # global surface        = union(box1, box2) - inter(box1, box2)
            global_surface          = surface_box1 + surface_box2 - bonding_box_surface 


            # compute iou values 
            iou                     = (bonding_box_surface) / global_surface 


            return iou
    '''
    
    corner     = '''

        def yolo_boxes_to_corners(
            box_xy : any, 
            box_wh : any
            ):

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
    '''
    
    eval_      = '''
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
    '''
    
    model      = '''
        import tensorflow as tf 

        def custom_function(x, y):
            return x ** 2 

        def yolo_model_for_CV(
            input_shape : tuple, 
            num_classes : int = 2, 
            num_anchors : int = 1
            ):
            
            input_layer = tf.keras.layers.Input(shape=input_shape, name='input_layer')

            # Feature extraction layers
            x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(input_layer)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

            x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

            x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

            x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

            x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        
            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)

            x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same")(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)

            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.LeakyReLU()(x)
            
            x = tf.keras.layers.BatchNormalization(axis=3)(x)

            xx = tf.keras.layers.Lambda(custom_function)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Concatenate()([xx, x])

            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.LeakyReLU()(x)

            # Detection layers
            x = tf.keras.layers.Conv2D(425, (1, 1), strides=(1, 1), padding='same', activation=tf.keras.layers.LeakyReLU())(x)

            model = tf.keras.models.Model(inputs=input_layer, outputs=x, name='yolo_model')

            return model

    '''
    
    string = None 

    if name == "yolo filter boxes" : string = filter_box
    if name == "yolo-non-max suppression" : string = non_max
    if name == "IoU" : string = iou
    if name == "yolo boxes to corners" : string= corner
    if name == "yolo evaluation" : string= eval_
    if name == "yolo model" : string = model

    return string