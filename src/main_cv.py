from data_preprocessing import ptbdb_data_scaled, load_ptbdb_data, \
    split_and_scale_ptbdb_data, load_mitbih_data, split_and_scale_mitbih_data, mitbih_data_scaled
from autoencoder import create_autoencoder_model, get_encoder
from data_processing import data_reconstraction
from src.classification_model import train_cv_classifier
from src.visualization import plot_cross_performance, \
    plot_cross_training_history, plot_cross_confusion_matrices, plot_cv_attention_heads, plot_cv_highlighted_ecg_parts
from visualization import (plot_reconstruction_comparison)
def main():
    #Load and preprocess MIT-BIH data
    train_data, test_data = load_mitbih_data()
    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_mitbih_data(train_data, test_data)

    """#Load and preprocess PTBDB data
    data = load_ptbdb_data()
    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_ptbdb_data(data)"""

    # Create and train autoencoder
    input_shape = (X_train.shape[1], 1)
    autoencoder = create_autoencoder_model(input_shape)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    #Visualise reconstructed samples
    reconstructed_data, X_test_original, mae_per_sample = data_reconstraction(X_test, autoencoder, scaler)
    plot_reconstruction_comparison(X_test_original, reconstructed_data, mae_per_sample, "../output/reconstructed_data.png")

    # Get encoded representations
    encoder = get_encoder(autoencoder)
    """X_all, Y_all = mitbih_data_scaled(scaler)
    encoded_all = encoder.predict(X_all)"""
    X_all, Y_all = ptbdb_data_scaled(scaler)
    encoded_all = encoder.predict(X_all)

    history = train_cv_classifier(encoded_all, Y_all, X_all, scaler, 50)
    all_accuracies = history[0]
    all_f1s = history[1]
    fold_histories = history[2]
    all_confusion_matrices = history[3]
    flat_attention_weights = history[4]
    all_fold_test_samples = history[5]
    flat_signals = history[6]

    plot_cross_performance(all_accuracies, all_f1s,"../output/cross_validation_accuracy_f1.png")
    plot_cross_training_history(fold_histories, "../output/cross_performance_history.png")

    sample_idx = 1700

    plot_cv_attention_heads(flat_attention_weights, sample_idx, "../output/heads.png")
    plot_cv_highlighted_ecg_parts(flat_signals, flat_attention_weights, 40, 0.1,"../output/highlighted_ecg_parts.png")
    plot_cross_confusion_matrices(all_confusion_matrices, Y_all,"../output/cross_confusion_matrices.png")

if __name__ == "__main__":
    main()