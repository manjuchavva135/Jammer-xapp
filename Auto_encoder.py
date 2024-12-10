import json
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

def load_json_files(file_paths):
    data_entries = []
    for path in file_paths:
        with open(path, 'r') as file:
            data_entries.extend(json.load(file))
    return data_entries

def extract_features(data):
    rows = []
    for entry in data:
        if entry['type'] == 'metrics':
            for cell in entry['cell_list']:
                for ue in cell['cell_container'].get('ue_list', []):
                    ue_data = ue['ue_container']
                    rows.append({
                        'timestamp': entry['timestamp'],
                        'ue_rnti': ue_data.get('ue_rnti', None),
                        'dl_cqi': ue_data.get('dl_cqi', None),
                        'dl_mcs': ue_data.get('dl_mcs', None),
                        'dl_bitrate': ue_data.get('dl_bitrate', None),
                        'dl_bler': ue_data.get('dl_bler', None),
                        'ul_snr': ue_data.get('ul_snr', None),
                        'ul_mcs': ue_data.get('ul_mcs', None),
                        'ul_bitrate': ue_data.get('ul_bitrate', None),
                        'ul_bler': ue_data.get('ul_bler', None),
                        'ul_phr': ue_data.get('ul_phr', None),
                        'ul_bsr': ue_data.get('ul_bsr', None),
                        'nof_rach': cell['cell_container'].get('nof_rach', None)
                    })
    return rows

def main():
    # Define file paths
    jammer_file_paths = ['/home/manju/Downloads/kpm-data/kpm-data2/jammer1.json', '/home/manju/Downloads/kpm-data/kpm-data2/jammer2.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer3.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer4.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer5.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer6.json']
    clean_file_paths = ['/home/manju/Downloads/kpm-data/kpm-data2/clean1.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean2.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean3.json','/home/manju/Downloads/kpm-data/kpm-data2/clean4.json']

    # Load data from clean files for training
    clean_data = load_json_files(clean_file_paths)
    clean_features = extract_features(clean_data)
    clean_df = pd.DataFrame(clean_features)

    # Load data from jammer files for testing
    jammer_data = load_json_files(jammer_file_paths)
    jammer_features = extract_features(jammer_data)
    jammer_df = pd.DataFrame(jammer_features)

    # Save extracted features to CSV files for training and testing
    clean_df.to_csv('training_data.csv', index=False)
    jammer_df.to_csv('testing_data.csv', index=False)

    print("Training and testing data have been saved to 'training_data.csv' and 'testing_data.csv' respectively.")

    # Load training and testing data
    training_data = pd.read_csv('training_data.csv')
    testing_data = pd.read_csv('testing_data.csv')

    # Drop timestamp and ue_rnti columns for model training
    training_data = training_data.drop(['timestamp', 'ue_rnti'], axis=1)
    testing_data = testing_data.drop(['timestamp', 'ue_rnti'], axis=1)

    # Replace NaN values with 0
    training_data = training_data.fillna(0)
    testing_data = testing_data.fillna(0)

    # Standardize the data
    scaler = StandardScaler()
    training_data_scaled = scaler.fit_transform(training_data)
    testing_data_scaled = scaler.transform(testing_data)

    # Define the autoencoder model
    input_dim = training_data_scaled.shape[1]
    encoding_dim = int(input_dim / 2)

    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,),
                          activity_regularizer=regularizers.l1(10e-5)))
    autoencoder.add(Dense(input_dim, activation="relu"))
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(training_data_scaled, training_data_scaled,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.2)

    # Evaluate the autoencoder on test data
    predictions = autoencoder.predict(testing_data_scaled)
    reconstruction_error = np.mean(np.power(testing_data_scaled - predictions, 2), axis=1)

    # Save reconstruction error to CSV
    pd.DataFrame(reconstruction_error, columns=['reconstruction_error']).to_csv('reconstruction_error.csv', index=False)
    print("Reconstruction error has been saved to 'reconstruction_error.csv'.")
    # Plot reconstruction error for analysis
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error, bins=50, alpha=0.7, color='b', label='Reconstruction Error')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reconstruction Errors')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Determine threshold for anomaly detection (e.g., using percentile)
    threshold = np.percentile(reconstruction_error, 96)
    print(f"Threshold for anomaly detection (95th percentile): {threshold}")

    # Label anomalies based on reconstruction error
    anomalies = reconstruction_error > threshold
    testing_data['is_anomaly'] = anomalies
    testing_data.to_csv('testing_data_with_anomalies.csv', index=False)
    print("Testing data with anomaly labels has been saved to 'testing_data_with_anomalies.csv'.")

if __name__ == "__main__":
    main()
