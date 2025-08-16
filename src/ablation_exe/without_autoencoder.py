from datetime import datetime
from src.processing.data_preprocessing import load_mitbih_data, split_and_scale_mitbih_data, mitbih_data_scaled, \
    split_and_scale_ptbdb_data, load_ptbdb_data, ptbdb_data_scaled
from src.models.classification_model import train_cv_classifier
from src.utils.utils import execution_time
from src.utils.visualization import plot_cross_performance, \
    plot_cross_training_history, plot_cross_confusion_matrices, plot_cv_attention_heads, plot_cv_highlighted_ecg_parts
def main():

    start_time = datetime.now()

    #Load and preprocess MIT-BIH data
    train_data, test_data = load_mitbih_data()
    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_mitbih_data(train_data, test_data)

    """#Load and preprocess PTBDB data
    data = load_ptbdb_data()
    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_ptbdb_data(data)"""


    X_all, Y_all = mitbih_data_scaled(scaler)

    #X_all, Y_all = ptbdb_data_scaled(scaler)

    history = train_cv_classifier(X_all, X_all, Y_all , scaler, 50)
    all_accuracies = history[0]
    all_f1s = history[1]
    fold_histories = history[2]
    all_confusion_matrices = history[3]
    flat_attention_weights = history[4]
    flat_signals = history[5]

    plot_cross_performance(all_accuracies, all_f1s,"../../output/cross_validation_accuracy_f1.png")
    plot_cross_training_history(fold_histories, "../../output/cross_performance_history.png")

    sample_idx = 1700

    plot_cv_attention_heads(flat_attention_weights, sample_idx, "../../output/heads.png")
    plot_cv_highlighted_ecg_parts(flat_signals, flat_attention_weights, 40, 0.1,
                                  "../../output/highlighted_ecg_parts.png")
    plot_cross_confusion_matrices(all_confusion_matrices, Y_all,"../../output/cross_confusion_matrices.png")

    end_time = datetime.now()

    print("Execution time: ", execution_time(start_time, end_time))

if __name__ == "__main__":
    main()