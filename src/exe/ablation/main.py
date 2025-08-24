from datetime import datetime
import pandas as pd
from src.models.classification_models_pl import train_cv_classifier
from src.models.autoencoder import create_autoencoder_model, get_encoder
from src.processing.data_processing import data_reconstraction
from src.processing.data_preprocessing_pl import add_synthetic_patient_id_ptbdb, \
    split_by_groupshuffle_and_scale, ptbdb_data_scaled_pl, add_synthetic_patient_id, load_mitbih_data_pl, \
    split_and_scale_mitbih_pl, mitbih_data_scaled_pl
from src.utils.utils import execution_time
from src.utils.visualization import plot_ablation_cv_performance, plot_ablation_cv_training_history, \
    plot_ablation_cv_confusion_matrices, plot_cross_training_history, plot_cv_attention_heads, \
    plot_cv_highlighted_ecg_parts
from src.utils.visualization import (plot_reconstruction_comparison)
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



    # Create and train autoencoder
    input_shape = (X_train.shape[1], 1)
    autoencoder = create_autoencoder_model(input_shape)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    #Visualise reconstructed samples
    reconstructed_data, X_test_original, mae_per_sample = data_reconstraction(X_test, autoencoder, scaler)
    plot_reconstruction_comparison(X_test_original, reconstructed_data, mae_per_sample, "../../../output/reconstructed_data.png")

    # Get encoded representations
    encoder = get_encoder(autoencoder)

    encoded_all = encoder.predict(X_all)

    model_name = None

    history = train_cv_classifier(encoded_all, X_all, Y_all, scaler, 50, patient_ids, 10, model_name)
    all_accuracies = history[0]
    all_f1s = history[1]
    fold_histories = history[2]
    all_confusion_matrices = history[3]

    plot_ablation_cv_performance(all_accuracies, all_f1s,"../../../output/cross_validation_accuracy_f1.png")
    #plot_ablation_cv_training_history(fold_histories, "../../../output/cross_performance_history.png")
    plot_cross_training_history(fold_histories, "../../../output/cross_performance_history.png")
    plot_ablation_cv_confusion_matrices(all_confusion_matrices, Y_all,"../../../output/cross_confusion_matrices.png")

    if (not model_name or model_name == "transformer"):
        sample_idx = 1700
        flat_attention_weights = history[4]
        flat_signals = history[5]
        plot_cv_attention_heads(flat_attention_weights, sample_idx, "../../../output/heads.png")
        plot_cv_highlighted_ecg_parts(flat_signals, flat_attention_weights, sample_idx, 0.1,
                                      "../../../output/highlighted_ecg_parts.png")

    end_time = datetime.now()

    print("Execution time: ", execution_time(start_time, end_time))

if __name__ == "__main__":
    main()