import tensorflow as tf
import keras.backend as K
import numpy as np

def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes

def yolo_loss(self,
            args,
            anchors,
            num_classes,
            rescore_confidence=False,
            print_loss=False):
        
        """YOLO localization loss function.

        Parameters
        ----------
        yolo_output : tensor
            Final convolutional layer features.

        true_boxes : tensor
            Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
            containing box x_center, y_center, width, height, and class.

        detectors_mask : array
            0/1 mask for detector positions where there is a matching ground truth.

        matching_true_boxes : array
            Corresponding ground truth boxes for positive detector positions.
            Already adjusted for conv height and width.

        anchors : tensor
            Anchor boxes for model.

        num_classes : int
            Number of object classes.

        rescore_confidence : bool, default=False
            If true then set confidence target to IOU of best predicted box with
            the closest matching ground truth box.

        print_loss : bool, default=False
            If True then use a tf.Print() to print the loss components.

        Returns
        -------
        mean_loss : float
            mean localization loss across minibatch
        """
        (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
        num_anchors = len(anchors)
        object_scale = 5
        no_object_scale = 1
        class_scale = 1
        coordinates_scale = 1

        pred_xy, pred_wh, pred_confidence, pred_class_prob = self.yolo_head(
            yolo_output, anchors, num_classes)

        # Unadjusted box predictions for loss.
        # TODO: Remove extra computation shared with yolo_head.
        yolo_output_shape = K.shape(yolo_output)
        feats = K.reshape(yolo_output, [
            -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
            num_classes + 5
        ])
        pred_boxes = K.concatenate(
            (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

        # TODO: Adjust predictions by image width/height for non-square images?
        # IOUs may be off due to different aspect ratio.

        # Expand pred x,y,w,h to allow comparison with ground truth.
        # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
        pred_xy = K.expand_dims(pred_xy, 4)
        pred_wh = K.expand_dims(pred_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        true_boxes_shape = K.shape(true_boxes)

        # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
        true_boxes = K.reshape(true_boxes, [
            true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
        ])
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        # Find IOU of each predicted box with each ground truth box.
        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        intersect_mins = K.maximum(pred_mins, true_mins)
        intersect_maxes = K.minimum(pred_maxes, true_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        # Best IOUs for each location.
        best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
        best_ious = K.expand_dims(best_ious)

        # A detector has found an object if IOU > thresh for some true box.
        object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

        # TODO: Darknet region training includes extra coordinate loss for early
        # training steps to encourage predictions to match anchor priors.

        # Determine confidence weights from object and no_object weights.
        # NOTE: YOLO does not use binary cross-entropy here.
        no_object_weights = (no_object_scale * (1 - object_detections) *
                            (1 - detectors_mask))
        no_objects_loss = no_object_weights * K.square(-pred_confidence)

        if rescore_confidence:
            objects_loss = (object_scale * detectors_mask *
                            K.square(best_ious - pred_confidence))
        else:
            objects_loss = (object_scale * detectors_mask *
                            K.square(1 - pred_confidence))
        confidence_loss = objects_loss + no_objects_loss

        # Classification loss for matching detections.
        # NOTE: YOLO does not use categorical cross-entropy loss here.

        matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
        matching_classes = K.one_hot(matching_classes, num_classes)
        classification_loss = (class_scale * detectors_mask *
                            K.square(matching_classes - pred_class_prob))

        # Coordinate loss for matching detection boxes.
        matching_boxes = matching_true_boxes[..., 0:4]
        coordinates_loss = (coordinates_scale * detectors_mask *
                            K.square(matching_boxes - pred_boxes))

        confidence_loss_sum = K.sum(confidence_loss)
        classification_loss_sum = K.sum(classification_loss)
        coordinates_loss_sum = K.sum(coordinates_loss)
        total_loss = 0.5 * (
            confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
        if print_loss:
            total_loss = tf.Print(
                total_loss, [
                    total_loss, confidence_loss_sum, classification_loss_sum,
                    coordinates_loss_sum
                ],
                message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

        return total_loss

# Fonction de perte YOLOv5 personnalisée
def yolo_loss(y_true, y_pred):
    # y_true : Les vérités terrain (annotations)
    # y_pred : Les prédictions du modèle

    # Divisez y_pred en ses composantes : confiance, coordonnées des boîtes, classes
    pred_confidence     = y_pred[..., 0]  # Confidence
    pred_boxes          = y_pred[..., 1:5]     # Coordonnées des boîtes (x, y, largeur, hauteur)
    pred_classes        = y_pred[..., 5:]    # Probabilités de classe

    # Divisez y_true en ses composantes correspondantes
    true_confidence     = y_true[..., 0]  # Confidence
    true_boxes          = y_true[..., 1:5]     # Coordonnées des boîtes (x, y, largeur, hauteur)
    true_classes        = y_true[..., 5:]    # Probabilités de classe

    # Calcul de la perte de confiance (confidence loss)
    confidence_loss     = tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence)

    # Calcul de la perte de boîte (bounding box loss) avec la perte L1
    box_loss            = tf.keras.losses.mean_squared_error(true_boxes, pred_boxes) #mean_absolute_error

    # Calcul de la perte de classe (class loss) avec la perte de log-vraisemblance
    class_loss          = tf.keras.losses.categorical_crossentropy(true_classes, pred_classes)

    # Summez les trois composantes de perte
    total_loss          = confidence_loss + box_loss + class_loss

    return total_loss


def loss_function(y_true, y_pred):

    """
    y_pred.shape = (4, )
    y_true.shape = (4, )

    * y_pred is the prediction
    * y_true is the expected values 
 
    x_p, y_p, w_p, h_p = y_pred 
    x_t, y_t, w_t, h_t = y_true
    here the loss is computed as 

    loss = (x_t - x_t)^2 + (y_t - y_p)^2 + (|w_t|^(.5) - |w_p|^(.5))^2 + (|h_t|^(.5) - |h_p|^(.5))^2

    here we just have few classes to predict

    """
    return  K.abs(y_true[ :, 0]-y_pred[ :, 0]) +\
            K.abs(y_true[ :, 1]-y_pred[ :, 1]) +\
            K.square( 
                    K.sqrt(K.abs(y_true[ :, 2])) -\
                    K.sqrt(K.abs(y_pred[ :, 2]))
                    ) +\
            K.square(
                    K.sqrt(K.abs(y_true[ :, 3])) -\
                    K.sqrt(K.abs(y_pred[ :, 3]))
                    )
