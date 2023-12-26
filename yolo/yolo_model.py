import tensorflow as tf
from keras import backend 
import functools
from functools import partial
from functools import reduce
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from keras.layers import Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from yolo.yolo_head import yolo_head

"""Darknet19 Model Defined in Keras."""
# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')

def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def space_to_depth(x, block_size=2):
    return tf.nn.space_to_depth(x, block_size=block_size)

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
 
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def bottleneck_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))

def bottleneck_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))

def darknet_body():
    """Generate first 18 conv layers of Darknet-19."""
    return compose(
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(),
        bottleneck_block(128, 64),
        MaxPooling2D(),
        bottleneck_block(256, 128),
        MaxPooling2D(),
        bottleneck_x2_block(512, 256),
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512))

def darknet19(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    body = darknet_body()(inputs)
    logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)

    return Model(inputs, logits)

def yolo_body(inputs, num_anchors, num_classes):

    """Create YOLO_V2 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body()(inputs))
    conv20  = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(1024, (3, 3)))(darknet.output)

    conv13 = darknet.layers[43].output
    conv21 = DarknetConv2D_BN_Leaky(64, (1, 1))(conv13)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv21_reshaped = Lambda(
        space_to_depth,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv21)

    x = Concatenate([conv21_reshaped, conv20])
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)

    return Model(inputs, x)

def yolo(inputs, anchors, num_classes):
    """Generate a complete YOLO_v2 localization model."""
    num_anchors = len(anchors)
    body = yolo_body(inputs, num_anchors, num_classes)
    outputs = yolo_head(body.output, anchors, num_classes)
    return outputs

def create_yolo_model_for_cmputer_vision(input_shape : tuple, num_anchors : int, num_classes : int):
    input_1 = tf.keras.layers.Input(shape=input_shape, name='input_1')

    # Feature extraction layers
    conv2d                  = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same', name='conv2d')(input_1)
    batch_normalization     = tf.keras.layers.BatchNormalization(axis=3, name='batch_normalization')(conv2d)
    leaky_re_lu             = tf.keras.layers.LeakyReLU(name="leaky_re_lu")(batch_normalization)
    max_pooling2d           = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d")(leaky_re_lu)

    conv2d_1                = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name="conv2d_1")(max_pooling2d)
    batch_normalization_1   = tf.keras.layers.BatchNormalization(axis=3,name="batch_normalization_1")(conv2d_1)
    leaky_re_lu_1           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_1")(batch_normalization_1)
    max_pooling2d_1         = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1")(leaky_re_lu_1)
    
    conv2d_2                = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name="conv2d_2")(max_pooling2d_1)
    batch_normalization_2   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_2")(conv2d_2)
    leaky_re_lu_2           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_2")(batch_normalization_2)
    
    conv2d_3                = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name="conv2d_3")(leaky_re_lu_2)
    batch_normalization_3   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_3")(conv2d_3)
    leaky_re_lu_3           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_3")(batch_normalization_3)
    
    conv2d_4                = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name="conv2d_4")(leaky_re_lu_3)
    batch_normalization_4   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_4")(conv2d_4)
    leaky_re_lu_4           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_4")(batch_normalization_4)
    max_pooling2d_2         = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2")(leaky_re_lu_4)
    
    conv2d_5                = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name="conv2d_5")(max_pooling2d_2)
    batch_normalization_5   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_5")(conv2d_5)
    leaky_re_lu_5           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_5")(batch_normalization_5)
    
    conv2d_6                = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name="conv2d_6")(leaky_re_lu_5)
    batch_normalization_6   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_6")(conv2d_6)
    leaky_re_lu_6           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_6")(batch_normalization_6)
    
    conv2d_7                = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name="conv2d_7")(leaky_re_lu_6)
    batch_normalization_7   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_7")(conv2d_7)
    leaky_re_lu_7           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_7")(batch_normalization_7)
    max_pooling2d_3         = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3")(leaky_re_lu_7)
    
    conv2d_8                = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name="conv2d_8")(max_pooling2d_3)
    batch_normalization_8   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_8")(conv2d_8)
    leaky_re_lu_8           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_8")(batch_normalization_8)
    
    conv2d_9                = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name="conv2d_9")(leaky_re_lu_8)
    batch_normalization_9   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_9")(conv2d_9)
    leaky_re_lu_9           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_9")(batch_normalization_9)
    
    conv2d_10                = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name="conv2d_10")(leaky_re_lu_9)
    batch_normalization_10   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_10")(conv2d_10)
    leaky_re_lu_10           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_10")(batch_normalization_10)
    #37
    conv2d_11                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_11")(leaky_re_lu_10)
    batch_normalization_11   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_11")(conv2d_11)
    leaky_re_lu_11           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_11")(batch_normalization_11)
    
    conv2d_12                = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name="conv2d_12")(leaky_re_lu_11)
    batch_normalization_12   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_12")(conv2d_12)
    leaky_re_lu_12           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_12")(batch_normalization_12)
    max_pooling2d_4          = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_4")(leaky_re_lu_12)
    
    conv2d_13                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_13")(max_pooling2d_4)
    batch_normalization_13   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_13")(conv2d_13)
    leaky_re_lu_13           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_13")(batch_normalization_13)
    
    conv2d_14                = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name="conv2d_14")(leaky_re_lu_13)
    batch_normalization_14   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_14")(conv2d_14)
    leaky_re_lu_14           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_14")(batch_normalization_14)
    
    conv2d_15                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_15")(leaky_re_lu_14)
    batch_normalization_15   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_15")(conv2d_15)
    leaky_re_lu_15           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_15")(batch_normalization_15)

    conv2d_16                = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name="conv2d_16")(leaky_re_lu_15)
    batch_normalization_16   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_16")(conv2d_16)
    leaky_re_lu_16           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_16")(batch_normalization_16)

    conv2d_17                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_17")(leaky_re_lu_16)
    batch_normalization_17   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_17")(conv2d_17)
    leaky_re_lu_17           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_17")(batch_normalization_17)

    conv2d_18                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_18")(leaky_re_lu_17)
    batch_normalization_18   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_18")(conv2d_18)

    conv2d_20                = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name="conv2d_20")(leaky_re_lu_12)
    leaky_re_lu_18           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_18")(batch_normalization_18)
    batch_normalization_20   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_20")(conv2d_20)

    conv2d_19                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_19")(leaky_re_lu_18)
    leaky_re_lu_20           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_20")(batch_normalization_20)
    batch_normalization_19   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_19")(conv2d_19)
    
    space_to_depth_x2        = tf.keras.layers.Lambda(space_to_depth, output_shape=space_to_depth_x2_output_shape,
                                                name="space_to_depth_x2")(leaky_re_lu_20)
    leaky_re_lu_19           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_19")(batch_normalization_19)
    concatenated             = tf.keras.layers.Concatenate(name="concatenate")([space_to_depth_x2, leaky_re_lu_19])

    conv2d_21                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_21")(concatenated)
    batch_normalization_21   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_21")(conv2d_21)
    leaky_re_lu_21           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_21")(batch_normalization_21)
       
    # Detection layers
    conv2d_22 = tf.keras.layers.Conv2D(num_anchors * (num_classes + 5), (1, 1), 
                        strides=(1, 1), padding='same',  name="conv2d_22")(leaky_re_lu_21)
    
    model = tf.keras.models.Model(inputs=input_1, outputs=conv2d_22, name='yolo_model')
    
    return model
