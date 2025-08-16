from datetime import datetime

from src.processing.data_preprocessing import load_mitbih_data, split_and_scale_mitbih_data, load_ptbdb_data, \
    split_and_scale_ptbdb_data
from src.models.autoencoder import create_autoencoder_model, get_encoder
from src.processing.data_processing import data_reconstraction
from src.models.classification_model import train_classifier
from src.utils.visualization import (plot_reconstruction_comparison, plot_training_history, plot_roc_curve,
                         plot_confusion_matrix, plot_attention_heads, plot_highlighted_ecg_parts)
from src.utils.utils import calculate_metrics, execution_time


def main():

    start_time = datetime.now()
    # Load and preprocess MIT-BIH data
    """train_data, test_data = load_mitbih_data()
    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_mitbih_data(train_data, test_data)"""

    # Load and preprocess PTBDB data
    data = load_ptbdb_data()
    X_train, X_test, Y_train, Y_test, scaler = split_and_scale_ptbdb_data(data)

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
    encoded_train = encoder.predict(X_train)
    encoded_test = encoder.predict(X_test)

    # Create and train classifier

    classifier, history = train_classifier(encoded_train, encoded_test, Y_train, Y_test, epochs=50)

    # Evaluate and visualize results
    predictions, attention_weights = classifier.predict(encoded_test)
    metrics = calculate_metrics(Y_test, predictions)
    #print_metrics(metrics)

    # Generate visualizations
    plot_training_history(history, "../../output/accuracy_loss.png")
    plot_confusion_matrix(Y_test, predictions, "../../output/confusion_matrix.png")
    plot_roc_curve(Y_test, predictions, "../../output/roc_curve.png")
    plot_attention_heads(attention_weights, "../../output/heads.png")
    plot_highlighted_ecg_parts(X_test_original, attention_weights,40,"../../output/highlighted_ecg_parts.png")

    end_time = datetime.now()

    print("Execution time: ", execution_time(start_time, end_time))

if __name__ == "__main__":
    main()