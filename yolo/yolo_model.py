import tensorflow as tf
from keras import backend 

def space_to_depth(x, block_size):
    return tf.nn.space_to_depth(x, block_size=block_size)

def create_yolo_model_for_cmputer_vision(input_shape : tuple):
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
    
    space_to_depth_x2        = tf.keras.layers.Lambda(lambda x: space_to_depth(x, block_size=2), name="space_to_depth_x2")(leaky_re_lu_20)
    leaky_re_lu_19           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_19")(batch_normalization_19)
    concatenated             = tf.keras.layers.Concatenate(name="concatenate")([space_to_depth_x2, leaky_re_lu_19])

    conv2d_21                = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name="conv2d_21")(concatenated)
    batch_normalization_21   = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_21")(conv2d_21)
    leaky_re_lu_21           = tf.keras.layers.LeakyReLU(name="leaky_re_lu_21")(batch_normalization_21)
       
    # Detection layers
    conv2d_22 = tf.keras.layers.Conv2D(425, (1, 1), 
                        strides=(1, 1), padding='same',  name="conv2d_22")(leaky_re_lu_21)
    
    model = tf.keras.models.Model(inputs=input_1, outputs=conv2d_22, name='yolo_model')
    
    return model
