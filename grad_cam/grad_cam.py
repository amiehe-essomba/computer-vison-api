import tensorflow as tf 
import cv2
import numpy as np
from skimage.transform import resize

def grad_cam(img_array, grad_model):
    # Calculez le gradient de la classe cible par rapport à l'activation de la dernière couche convolutive
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, None]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def guided_backprop(model, img, upsample_size=(608, 608)):
        model.trainable = True
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)
            tape.watch(inputs)
            outputs = model(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency

def deprocess_image(x):
    import keras.backend as K
  
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255*cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    
    new_img = 0.3*cam3 + 0.5*img
    
    return (new_img*255.0/new_img.max()).astype("uint8")

def CompGradcam(model, img,  alpha=0.4, shape = (608, 608)):
    import matplotlib.cm as cm
    from PIL import Image

    img     = resize(image=img, output_shape=shape)
    shape1  = img.shape 
    shape2  = (1,) + shape1

    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.layers[-10].output, model.output])

    img = img.reshape(shape2) #(1, 128, 128, 3)
    IMG = img.reshape(shape1).astype('float32') # (128, 128, 3)

    heatmap     = grad_cam(img, grad_model)
    img         = IMG.copy()
    # Load the original image
    img         = tf.keras.utils.img_to_array(img)
    img         = np.uint8(255 * img)
    # Rescale heatmap to a range 0-255
    heatmap     = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet         = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors  = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)


    _ = IMG.reshape(shape2) # (1, 128, 128, 3)
    gb = guided_backprop(model=grad_model, img=_, upsample_size=shape) # (128,128)

    guided_gradcam = deprocess_image(gb * jet_heatmap)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)


    return superimposed_img, guided_gradcam

