import tensorflow as tf
import keras.backend as K

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
    return  K.square(y_true[ :, 0]-y_pred[ :, 0]) +\
            K.square(y_true[ :, 1]-y_pred[ :, 1]) +\
            K.square( 
                    K.sqrt(K.abs(y_true[ :, 2])) -\
                    K.sqrt(K.abs(y_pred[ :, 2]))
                    ) +\
            K.square(
                    K.sqrt(K.abs(y_true[ :, 3])) -\
                    K.sqrt(K.abs(y_pred[ :, 3]))
                    )
