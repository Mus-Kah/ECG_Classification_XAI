import tensorflow as tf

def create_autoencoder_model(input_shape):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    encoded = x

    x = tf.keras.layers.LSTM(16, return_sequences=True)(encoded)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    decoded = tf.keras.layers.UpSampling1D(2)(x)

    autoencoder = tf.keras.Model(encoder_inputs, decoded)
    return autoencoder

def get_encoder(autoencoder):
    return tf.keras.Model(autoencoder.input, autoencoder.layers[-7].output)