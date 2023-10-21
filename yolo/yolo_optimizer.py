import tensorflow as tf

def optimizer(learning_rate : float = 0.001):
    # Taux d'apprentissage
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8, use_ema=True)

    return opt