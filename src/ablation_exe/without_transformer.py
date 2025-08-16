from datetime import datetime

from src.models.ablation_classification_models import train_ablation_cv_classifier
from src.processing.data_preprocessing import load_mitbih_data, split_and_scale_mitbih_data, mitbih_data_scaled, \
    split_and_scale_ptbdb_data, load_ptbdb_data, ptbdb_data_scaled
from src.models.autoencoder import create_autoencoder_model, get_encoder
from src.processing.data_processing import data_reconstraction
from src.utils.utils import execution_time
from src.utils.visualization import plot_ablation_cv_performance, plot_ablation_cv_training_history, plot_ablation_cv_confusion_matrices
from src.utils.visualization import (plot_reconstruction_comparison)
def main():

    start_time = datetime.now()

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
    plot_reconstruction_comparison(X_test_original, reconstructed_data, mae_per_sample, "../../output/reconstructed_data.png")

    # Get encoded representations
    encoder = get_encoder(autoencoder)

    X_all, Y_all = mitbih_data_scaled(scaler)
    encoded_all = encoder.predict(X_all)

    """X_all, Y_all = ptbdb_data_scaled(scaler)
    encoded_all = encoder.predict(X_all)"""

    history = train_ablation_cv_classifier(encoded_all, X_all, Y_all , scaler, 50)
    all_accuracies = history[0]
    all_f1s = history[1]
    fold_histories = history[2]
    all_confusion_matrices = history[3]

    plot_ablation_cv_performance(all_accuracies, all_f1s,"../../output/cross_validation_accuracy_f1.png")
    plot_ablation_cv_training_history(fold_histories, "../../output/cross_performance_history.png")
    plot_ablation_cv_confusion_matrices(all_confusion_matrices, Y_all,"../../output/cross_confusion_matrices.png")

    end_time = datetime.now()

    print("Execution time: ", execution_time(start_time, end_time))

if __name__ == "__main__":
    main()