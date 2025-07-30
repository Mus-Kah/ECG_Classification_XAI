from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def calculate_metrics(Y_test, predictions):
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(Y_test, predicted_classes)
    precision = precision_score(Y_test, predicted_classes, average='weighted')
    recall = recall_score(Y_test, predicted_classes, average='weighted')
    f1 = f1_score(Y_test, predicted_classes, average='weighted')
    conf_matrix = confusion_matrix(Y_test, predicted_classes)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }


def print_metrics(metrics):
    print(f'accuracy: {metrics["accuracy"]:.2f}')
    print(f'Precision: {metrics["precision"]:.2f}')
    print(f'Recall: {metrics["recall"]:.2f}')
    print(f'F1 Score: {metrics["f1"]:.2f}')
    print('Confusion Matrix:')
    print(metrics["confusion_matrix"])