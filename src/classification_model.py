import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from src.utils import calculate_metrics


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

def train_classifier(encoded_train, encoded_test, Y_train, Y_test):

    num_classes = len(pd.unique(Y_train))
    classifier = create_transformer_classifier_with_attention(encoded_train.shape[1:], num_classes)
    classifier.compile(
        optimizer='adam',
        loss=['sparse_categorical_crossentropy', None],
        metrics=[['accuracy'], None]
    )

    history = classifier.fit(
        encoded_train,
        Y_train,
        epochs=5,
        batch_size=32,
        validation_data=(encoded_test, Y_test)
    )

    return classifier, history

"""def train_cross_classifier(encoded_all, Y_all):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_precisions, all_recalls, all_f1s, all_accuracies, fold_histories = [], [], [], [], []
    all_attention_weights = []
    all_fold_test_samples = []
    all_conf_matrices = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(encoded_all, Y_all)):
        print(f"\n--- Fold {fold + 1} ---")

        X_train_fold, X_test_fold = encoded_all[train_idx], encoded_all[test_idx]
        Y_train_fold, Y_test_fold = Y_all[train_idx], Y_all[test_idx]

        # Create a fresh classifier
        classifier = create_transformer_classifier_with_attention(input_shape=encoded_all.shape[1:],
                                                                  num_classes=len(np.unique(Y_all)))

        classifier.compile(
            optimizer='adam',
            loss=['sparse_categorical_crossentropy', None],
            metrics=[['accuracy'], None]
        )

        history = classifier.fit(
            X_train_fold,
            Y_train_fold,
            epochs=5,
            batch_size=32,
            validation_data=(X_test_fold, Y_test_fold),
            verbose=0)

        predictions, attention_weights = classifier.predict(X_test_fold)

        metrics = calculate_metrics(Y_test_fold, predictions)
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        conf_matrix = metrics['confusion_matrix']

        print(f"Accuracy: {accuracy:.6f} | Precision: {precision:.6f} | Recall: {recall:.6f} | F1: {f1:.6f}")

        # Save attention weights and samples
        all_attention_weights.append(attention_weights)
        all_fold_test_samples.append(X_test_fold)

        #Flatten attention weights and fold test samples
        all_attention_weights = np.concatenate(all_attention_weights, axis=0)
        all_fold_test_samples = np.concatenate(all_fold_test_samples, axis=0)

        # Save metrics
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(accuracy)
        all_conf_matrices.append(conf_matrix)
        fold_histories.append(history)

        #print(history.history.keys())

    print("\n=== Average Performance Across 10 Folds ===")
    print(f"Avg Accuracy: {np.mean(all_accuracies):.6f}")
    print(f"Avg Precision: {np.mean(all_precisions):.6f}")
    print(f"Avg Recall: {np.mean(all_recalls):.6f}")
    print(f"Avg F1 Score: {np.mean(all_f1s):.6f}")


    return all_accuracies, all_f1s, fold_histories, all_conf_matrices, all_attention_weights, all_fold_test_samples
"""


def train_cv_classifier(encoded_all, Y_all, X_all_original, scaler, epochs):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_precisions, all_recalls, all_f1s, all_accuracies, fold_histories = [], [], [], [], []
    all_attention_weights = []
    all_fold_test_samples = []
    all_fold_test_signals = []
    all_conf_matrices = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(encoded_all, Y_all)):
        print(f"\n--- Fold {fold + 1} ---")

        X_train_fold, X_test_fold = encoded_all[train_idx], encoded_all[test_idx]
        Y_train_fold, Y_test_fold = Y_all[train_idx], Y_all[test_idx]

        X_test_original_fold = scaler.inverse_transform(
            X_all_original[test_idx].reshape(X_all_original[test_idx].shape[0], -1)
        )

        classifier = create_transformer_classifier_with_attention(
            input_shape=encoded_all.shape[1:],
            num_classes=len(np.unique(Y_all))
        )

        classifier.compile(
            optimizer='adam',
            loss=['sparse_categorical_crossentropy', None],
            metrics=[['accuracy'], None]
        )

        history = classifier.fit(
            X_train_fold,
            Y_train_fold,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test_fold, Y_test_fold),
            verbose=0
        )

        predictions, attention_weights = classifier.predict(X_test_fold)

        metrics = calculate_metrics(Y_test_fold, predictions)
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        conf_matrix = metrics['confusion_matrix']

        print(f"Accuracy: {accuracy:.6f} | Precision: {precision:.6f} | Recall: {recall:.6f} | F1: {f1:.6f}")

        all_attention_weights.append(attention_weights)
        all_fold_test_samples.append(X_test_fold)
        all_fold_test_signals.append(X_test_original_fold)

        # Save metrics
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(accuracy)
        all_conf_matrices.append(conf_matrix)
        fold_histories.append(history)

    # Flattening
    flat_attention_weights = np.concatenate(all_attention_weights, axis=0)
    flat_encoded_samples = np.concatenate(all_fold_test_samples, axis=0)
    flat_signals = np.concatenate(all_fold_test_signals, axis=0)

    print("\n=== Average Performance Across 10 Folds ===")
    print(f"Avg Accuracy: {np.mean(all_accuracies):.6f}")
    print(f"Avg Precision: {np.mean(all_precisions):.6f}")
    print(f"Avg Recall: {np.mean(all_recalls):.6f}")
    print(f"Avg F1 Score: {np.mean(all_f1s):.6f}")

    return (all_accuracies, all_f1s, fold_histories, all_conf_matrices,
            flat_attention_weights, flat_encoded_samples, flat_signals)
