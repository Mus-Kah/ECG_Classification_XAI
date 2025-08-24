from sklearn.model_selection import GroupKFold
import numpy as np
import tensorflow as tf
from src.utils.utils import calculate_metrics


def create_mlp_classifier(input_shape, num_classes):
    classifier_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(classifier_input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    classifier_output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    classifier = tf.keras.Model(inputs=classifier_input, outputs=classifier_output)

    return classifier

def create_lstm_classifier(input_shape, num_classes):
    classifier_input = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(64, return_sequences=True)(classifier_input)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    classifier_output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    classifier = tf.keras.Model(inputs=classifier_input, outputs=classifier_output)
    return classifier


def create_cnn_classifier(input_shape, num_classes):
    classifier_input = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(classifier_input)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    classifier_output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    classifier = tf.keras.Model(inputs=classifier_input, outputs=classifier_output)
    return classifier

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



model_names = {
    'cnn': "Convolutional Neural Network",
    'lstm': "Long-Short-Term Memory Network",
    'mlp': "Multi-Layer Perceptron",
    'transformer': "Multi-head Attention Network",
    None: "Multi-head Attention Network"
}
def train_cv_classifier(encoded_all, X_all_original, Y_all, scaler, epochs, patient_ids, n_splits=10, model_name=None):
    gkf = GroupKFold(n_splits=n_splits)

    all_precisions, all_recalls, all_f1s, all_accuracies, fold_histories = [], [], [], [], []
    all_attention_weights = []
    all_fold_test_samples = []
    all_fold_test_signals = []
    all_conf_matrices = []

    print("\nClassification model: ", model_names[model_name])

    for fold, (train_idx, test_idx) in enumerate(gkf.split(encoded_all, Y_all, groups=patient_ids)):
        print(f"\n--- Fold {fold + 1} ---")

        X_train_fold, X_test_fold = encoded_all[train_idx], encoded_all[test_idx]
        Y_train_fold, Y_test_fold = Y_all[train_idx], Y_all[test_idx]

        X_test_original_fold = scaler.inverse_transform(
            X_all_original[test_idx].reshape(X_all_original[test_idx].shape[0], -1)
        )
        match model_name:
            case 'cnn':
                loss = ['sparse_categorical_crossentropy']
                metrics = ['accuracy']
                classifier = create_cnn_classifier(
                    input_shape=encoded_all.shape[1:],
                    num_classes=len(np.unique(Y_all))
                )
            case 'lstm':
                loss = ['sparse_categorical_crossentropy']
                metrics = ['accuracy']
                classifier = create_lstm_classifier(
                    input_shape=encoded_all.shape[1:],
                    num_classes=len(np.unique(Y_all))
                )
            case 'mlp':
                loss = ['sparse_categorical_crossentropy']
                metrics = ['accuracy']
                classifier = create_mlp_classifier(
                    input_shape=encoded_all.shape[1:],
                    num_classes=len(np.unique(Y_all))
                )
            case _:
                loss = ['sparse_categorical_crossentropy', None]
                metrics = [['accuracy'], None]
                classifier = create_transformer_classifier_with_attention(
                    input_shape=encoded_all.shape[1:],
                    num_classes=len(np.unique(Y_all))
                )


        classifier.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )

        history = classifier.fit(
            X_train_fold,
            Y_train_fold,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test_fold, Y_test_fold),
            verbose=0
        )
        if (model_name == None or model_name == 'transformer'):
            predictions, attention_weights = classifier.predict(X_test_fold)
            all_attention_weights.append(attention_weights)
        else:
            predictions = classifier.predict(X_test_fold)

        metrics = calculate_metrics(Y_test_fold, predictions)
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        conf_matrix = metrics['confusion_matrix']

        print(f"Accuracy: {accuracy:.6f} | Precision: {precision:.6f} | Recall: {recall:.6f} | F1: {f1:.6f}")

        all_fold_test_samples.append(X_test_fold)
        all_fold_test_signals.append(X_test_original_fold)

        # Save metrics
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(accuracy)
        all_conf_matrices.append(conf_matrix)
        fold_histories.append(history)

    # Flattening to visualize XAI in the case of multi-head attention network
    if (model_name == None or model_name == 'transformer'):
        flat_attention_weights = np.concatenate(all_attention_weights, axis=0)
        flat_encoded_samples = np.concatenate(all_fold_test_samples, axis=0)
        flat_signals = np.concatenate(all_fold_test_signals, axis=0)

    print(f"\n=== Average Performance Across {n_splits} Folds ===")
    print(f"Avg Accuracy: {np.mean(all_accuracies):.6f}")
    print(f"Avg Precision: {np.mean(all_precisions):.6f}")
    print(f"Avg Recall: {np.mean(all_recalls):.6f}")
    print(f"Avg F1 Score: {np.mean(all_f1s):.6f}")

    if (model_name == None or model_name == 'transformer'):
        return (all_accuracies, all_f1s, fold_histories, all_conf_matrices,
            flat_attention_weights, flat_signals)
    return (all_accuracies, all_f1s, fold_histories, all_conf_matrices)