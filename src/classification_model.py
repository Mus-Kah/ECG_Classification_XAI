import tensorflow as tf

def create_transformer_classifier_with_attention(input_shape, num_classes):
    classifier_input = tf.keras.Input(shape=input_shape)

    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])
    attention_output, attention_scores = attention_layer(classifier_input, classifier_input, return_attention_scores=True)

    x = tf.keras.layers.LayerNormalization()(attention_output)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    classifier_output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    classifier = tf.keras.Model(inputs=classifier_input, outputs=[classifier_output, attention_scores])
    return classifier