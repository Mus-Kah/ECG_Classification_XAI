from datetime import datetime
import pandas as pd

from src.models.classification_models_pl import train_cv_classifier
from src.processing.data_preprocessing_pl import load_mitbih_data_pl, add_synthetic_patient_id, \
    split_and_scale_mitbih_pl, mitbih_data_scaled_pl, add_synthetic_patient_id_ptbdb, \
    split_by_groupshuffle_and_scale, ptbdb_data_scaled_pl
from src.utils.utils import execution_time
from src.utils.visualization import plot_cross_performance, \
    plot_cross_training_history, plot_cross_confusion_matrices, plot_cv_attention_heads, plot_cv_highlighted_ecg_parts
def main():

    start_time = datetime.now()

    # Load and preprocess MITBIH data
    train_data, test_data = load_mitbih_data_pl()
    train_with_pid, next_pid = add_synthetic_patient_id(train_data, segments_per_patient=20, start_patient_id=0)
    test_with_pid, _ = add_synthetic_patient_id(test_data, segments_per_patient=20, start_patient_id=next_pid)

    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_mitbih_pl(train_with_pid, test_with_pid)
    data_with_pid = pd.concat([train_with_pid, test_with_pid], ignore_index=True)
    X_all, Y_all = mitbih_data_scaled_pl(scaler, data_with_pid)
    patient_ids = data_with_pid["patient_id"].values

    """# Load and preprocess PTBDB data
    data, patient_ids = add_synthetic_patient_id_ptbdb(20)
    X_train, X_test, Y_train, Y_test, scaler = split_by_groupshuffle_and_scale(data)
    X_all, Y_all = ptbdb_data_scaled_pl(scaler)"""

    history = train_cv_classifier(X_all, X_all, Y_all, scaler, 50, patient_ids, 10)
    all_accuracies = history[0]
    all_f1s = history[1]
    fold_histories = history[2]
    all_confusion_matrices = history[3]
    flat_attention_weights = history[4]
    flat_signals = history[5]

    plot_cross_performance(all_accuracies, all_f1s,"../../../output/cross_validation_accuracy_f1.png")
    plot_cross_training_history(fold_histories, "../../../output/cross_performance_history.png")

    sample_idx = 1700

    plot_cv_attention_heads(flat_attention_weights, sample_idx, "../../../output/heads.png")
    plot_cv_highlighted_ecg_parts(flat_signals, flat_attention_weights, sample_idx, 0.1,
                                  "../../../output/highlighted_ecg_parts.png")
    plot_cross_confusion_matrices(all_confusion_matrices, Y_all,"../../../output/cross_confusion_matrices.png")

    end_time = datetime.now()

    print("Execution time: ", execution_time(start_time, end_time))

if __name__ == "__main__":
    main()