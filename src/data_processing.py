import numpy as np

def data_reconstraction(X_test, autoencoder, scaler):
    reconstructed_data = autoencoder.predict(X_test)
    reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1])
    X_test_original = X_test.reshape(X_test.shape[0], X_test.shape[1])
    reconstructed_data = scaler.inverse_transform(reconstructed_data)
    X_test_original = scaler.inverse_transform(X_test_original)
    mae_per_sample = np.mean(np.abs(reconstructed_data - X_test_original), axis=1)
    return reconstructed_data, X_test_original, mae_per_sample


