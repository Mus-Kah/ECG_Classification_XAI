import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d

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

    if num_classes > 2:
        if len(Y_test.shape) == 1:
            Y_test = label_binarize(Y_test, classes=range(num_classes))

        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('tab10', num_classes)  # Use a colormap for distinct colors
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(Y_test[:, i], class_probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})', color=colors(i))

    else:
        fpr, tpr, _ = roc_curve(Y_test, class_probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)

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



def plot_cv_attention_heads(flat_attention_weights, sample_idx=0, save_path=None):
    sample_attention_weights = flat_attention_weights[sample_idx]

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
    # plt.show()


def plot_cv_highlighted_ecg_parts(flat_signals, flat_attention_weights, sample_idx=0, threshold=0.1, save_path=None):

    ecg_signal = flat_signals[sample_idx]
    attention_scores = flat_attention_weights[sample_idx]

    # Aggregate across heads and take diagonal (self-attention focus)
    aggregated_attention = np.mean(attention_scores, axis=0)  # (seq_len, seq_len)
    diagonal_attention = np.diagonal(aggregated_attention)  # (seq_len,)
    diagonal_attention_normalized = (diagonal_attention - np.min(diagonal_attention)) / (
            np.max(diagonal_attention) - np.min(diagonal_attention)
    )

    # Interpolate to align with ECG signal length
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
        where=(diagonal_attention_resized > threshold),
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
    # plt.show()


def plot_cross_performance(all_accuracies, all_f1s, save_path=None):
    # Initialize the seaborn style
    sns.set(style="whitegrid")
    folds = np.arange(1, 11)

    # Define color palettes
    palette_acc = sns.color_palette("Blues", 10)
    palette_f1 = sns.color_palette("Reds", 10)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].bar(folds, all_accuracies, color=palette_acc)
    axs[0].axhline(np.mean(all_accuracies), linestyle='--', color='black', label=f"Mean: {np.mean(all_accuracies):.2f}")
    axs[0].set_title("Accuracy per Fold")
    axs[0].set_xlabel("Fold")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xticks(folds)
    axs[0].legend()

    axs[1].bar(folds, all_f1s, color=palette_f1)
    axs[1].axhline(np.mean(all_f1s), linestyle='--', color='black', label=f"Mean: {np.mean(all_f1s):.2f}")
    axs[1].set_title("F1 Score per Fold")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("F1 Score")
    axs[1].set_xticks(folds)
    axs[1].legend()

    plt.tight_layout()
    #plt.show()
    if save_path:
        plt.savefig(save_path)

def plot_cross_training_history(fold_histories, save_path=None):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for i, history in enumerate(fold_histories):
        ax = axs[i]
        # Find the train and validation accuracy keys for this history
        acc_key = next((k for k in history.history if k.endswith('_accuracy') and not k.startswith('val')), None)
        val_acc_key = next((k for k in history.history if k.startswith('val') and k.endswith('_accuracy')), None)

        if acc_key is None or val_acc_key is None:
            print(f"Accuracy keys not found for fold {i + 1}, history keys: {history.history.keys()}")
            continue

        epochs = range(1, len(history.history[acc_key]) + 1)

        ax.plot(epochs, history.history[acc_key], label='Train Acc', color='blue', marker='o')
        ax.plot(epochs, history.history[val_acc_key], label='Val Acc', color='green', marker='s')

        ax.plot(epochs, history.history['loss'], label='Train Loss', color='red', linestyle='--', marker='x')
        ax.plot(epochs, history.history['val_loss'], label='Val Loss', color='orange', linestyle='--', marker='^')

        ax.set_title(f'Fold {i + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def plot_cross_confusion_matrices(all_conf_matrices, Y_all, save_path=None):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    classes = np.unique(Y_all)

    for i, cm in enumerate(all_conf_matrices):
        ax = axs[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Fold {i + 1}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.tick_params(axis='both', which='both', length=0)

        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f'{val}', ha='center', va='center',
                    color='white' if val > cm.max() / 2 else 'black', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    def plot_cross_performance(all_accuracies, all_f1s, save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="whitegrid")
        folds = np.arange(1, len(all_accuracies) + 1)

        palette_acc = sns.color_palette("Blues", len(all_accuracies))
        palette_f1 = sns.color_palette("Reds", len(all_f1s))

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        axs[0].bar(folds, all_accuracies, color=palette_acc)
        axs[0].axhline(np.mean(all_accuracies), linestyle='--', color='black',
                       label=f"Mean: {np.mean(all_accuracies):.3f}")
        axs[0].set_title("Accuracy per Fold")
        axs[0].set_xlabel("Fold")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_xticks(folds)
        axs[0].legend()

        axs[1].bar(folds, all_f1s, color=palette_f1)
        axs[1].axhline(np.mean(all_f1s), linestyle='--', color='black', label=f"Mean: {np.mean(all_f1s):.3f}")
        axs[1].set_title("F1 Score per Fold")
        axs[1].set_xlabel("Fold")
        axs[1].set_ylabel("F1 Score")
        axs[1].set_xticks(folds)
        axs[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()


def plot_ablation_cv_performance(all_accuracies, all_f1s, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    folds = np.arange(1, len(all_accuracies)+1)

    palette_acc = sns.color_palette("Blues", len(all_accuracies))
    palette_f1 = sns.color_palette("Reds", len(all_f1s))

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].bar(folds, all_accuracies, color=palette_acc)
    axs[0].axhline(np.mean(all_accuracies), linestyle='--', color='black', label=f"Mean: {np.mean(all_accuracies):.3f}")
    axs[0].set_title("Accuracy per Fold")
    axs[0].set_xlabel("Fold")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xticks(folds)
    axs[0].legend()

    axs[1].bar(folds, all_f1s, color=palette_f1)
    axs[1].axhline(np.mean(all_f1s), linestyle='--', color='black', label=f"Mean: {np.mean(all_f1s):.3f}")
    axs[1].set_title("F1 Score per Fold")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("F1 Score")
    axs[1].set_xticks(folds)
    axs[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_ablation_cv_training_history(fold_histories, save_path=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for i, history in enumerate(fold_histories):
        ax = axs[i]

        # Keras standard keys
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
        loss_key = 'loss'
        val_loss_key = 'val_loss'

        # Check if keys exist, else warn
        missing_keys = [k for k in [acc_key, val_acc_key, loss_key, val_loss_key] if k not in history.history]
        if missing_keys:
            print(f"Missing keys for fold {i+1}: {missing_keys}. Available keys: {list(history.history.keys())}")
            continue

        epochs = range(1, len(history.history[acc_key]) + 1)

        ax.plot(epochs, history.history[acc_key], label='Train Acc', color='blue', marker='o')
        ax.plot(epochs, history.history[val_acc_key], label='Val Acc', color='green', marker='s')

        ax.plot(epochs, history.history[loss_key], label='Train Loss', color='red', linestyle='--', marker='x')
        ax.plot(epochs, history.history[val_loss_key], label='Val Loss', color='orange', linestyle='--', marker='^')

        ax.set_title(f'Fold {i + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_ablation_cv_confusion_matrices(all_conf_matrices, Y_all, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    n_folds = len(all_conf_matrices)
    n_rows = 2
    n_cols = int(np.ceil(n_folds / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 8))
    axs = axs.flatten()

    classes = np.unique(Y_all)

    for i, cm in enumerate(all_conf_matrices):
        ax = axs[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Fold {i + 1}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.tick_params(axis='both', which='both', length=0)

        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f'{val}', ha='center', va='center',
                    color='white' if val > cm.max() / 2 else 'black', fontsize=10)

    # Remove empty subplots if any
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()