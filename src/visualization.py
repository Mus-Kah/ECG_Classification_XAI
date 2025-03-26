import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_reconstruction_comparison(X_test_original, reconstructed_data, mae_per_sample, save_path=None):
    num_examples_to_plot = 4
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    for i in range(num_examples_to_plot):
        row = i // 2
        col = i % 2
        axs[row, col].plot(X_test_original[i], label='Original ECG')
        axs[row, col].plot(reconstructed_data[i], label='Reconstructed ECG')
        axs[row, col].set_title(f'Sample {i + 1} (MAE: {mae_per_sample[i]:.4f})')
        axs[row, col].set_xlabel('Time Steps')
        axs[row, col].set_ylabel('Amplitude')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()


def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['dense_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_dense_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()


def plot_confusion_matrix(Y_test, predictions, save_path=None):
    predicted_classes = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(Y_test, predicted_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test),
                square=True, annot_kws={"size": 29})

    plt.xlabel('Predicted Label', fontsize=26)
    plt.ylabel('True Label', fontsize=26)
    if save_path:
        plt.savefig(save_path)
    #plt.show()
"""
def plot_roc_curve(Y_test, class_probabilities, save_path=None):
    num_classes = class_probabilities.shape[1]

    # Multi-class classification
    if num_classes > 2:
        # One-hot encode Y_test if not already one-hot
        if len(Y_test.shape) == 1:
            Y_test = label_binarize(Y_test, classes=range(num_classes))

        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(Y_test[:, i], class_probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    # Binary classification
    else:
        fpr, tpr, _ = roc_curve(Y_test, class_probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    """


def plot_roc_curve(Y_test, class_probabilities, save_path=None, show_plot=True):

    Y_test = np.array(Y_test)
    class_probabilities = np.array(class_probabilities)

    num_classes = class_probabilities.shape[1]

    # Multi-class classification
    if num_classes > 2:
        # One-hot encode Y_test if not already one-hot
        if len(Y_test.shape) == 1:
            Y_test = label_binarize(Y_test, classes=range(num_classes))

        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('tab10', num_classes)  # Use a colormap for distinct colors
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(Y_test[:, i], class_probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})', color=colors(i))

    # Binary classification
    else:
        fpr, tpr, _ = roc_curve(Y_test, class_probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

    # Plot settings
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def plot_attention_heads(attention_weights, save_path=None):
    sample_idx = 0
    sample_attention_weights = attention_weights[sample_idx]

    num_heads = sample_attention_weights.shape[0]
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()

    for head in range(min(num_heads, rows * cols)):
        ax = axes[head]
        ax.set_title(f"Attention Head {head + 1}")
        cax = ax.imshow(sample_attention_weights[head], cmap='viridis')
        fig.colorbar(cax, ax=ax, label="Attention Weight")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Token Position")

    for j in range(num_heads, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def plot_highlighted_ecg_parts(X_test_original, attention_weights, sample_idx=0, save_path=None):
    ecg_signal = X_test_original[sample_idx]
    attention_scores = attention_weights[sample_idx]
    aggregated_attention = np.mean(attention_scores, axis=0)
    diagonal_attention = np.diagonal(aggregated_attention)
    diagonal_attention_normalized = (diagonal_attention - np.min(diagonal_attention)) / (
            np.max(diagonal_attention) - np.min(diagonal_attention)
    )
    interp_func = interp1d(
        np.linspace(0, len(ecg_signal), len(diagonal_attention_normalized)),
        diagonal_attention_normalized,
        kind="linear",
        fill_value="extrapolate"
    )
    diagonal_attention_resized = interp_func(np.arange(len(ecg_signal)))
    plt.figure(figsize=(15, 6))
    plt.plot(ecg_signal, label="ECG Signal", color="blue", linewidth=1.5)
    plt.fill_between(
        range(len(ecg_signal)),
        ecg_signal,
        where=(diagonal_attention_resized > 0.1),
        color="orange",
        alpha=0.5,
        label="Highlighted Attention",
    )
    plt.title("ECG Signal with Highlighted Attention Regions")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    #plt.show()