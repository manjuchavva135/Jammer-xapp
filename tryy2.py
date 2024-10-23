import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define your files, setting two files aside for testing
train_files = [
    '/home/manju/Downloads/kpm-data/kpm-data2/jammer1.json', '/home/manju/Downloads/kpm-data/kpm-data2/jammer2.json',
    '/home/manju/Downloads/kpm-data/kpm-data2/jammer3.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer4.json',
    '/home/manju/Downloads/kpm-data/kpm-data2/jammer5.json', 
    '/home/manju/Downloads/kpm-data/kpm-data2/clean1.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean2.json', 
    '/home/manju/Downloads/kpm-data/kpm-data2/clean3.json'
]

# Files for testing (one from each class)
test_files = [
    '/home/manju/Downloads/kpm-data/kpm-data2/jammer6.json',  # Jammer class
    '/home/manju/Downloads/kpm-data/kpm-data2/clean4.json'    # Clean class
]

def load_data_from_files(files):
    data = []
    labels = []
    for file in files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            for entry in json_data:
                if entry["type"] == "metrics":
                    # Extract basic features
                    nof_rach = entry["cell_list"][0]["cell_container"]["nof_rach"]
                    ue_list = entry["cell_list"][0]["cell_container"]["ue_list"]
                    
                    # Initialize aggregate metrics
                    num_ues = len(ue_list)
                    avg_dl_cqi = np.mean([ue.get("ue_container", {}).get("dl_cqi", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_ul_snr = np.mean([ue.get("ue_container", {}).get("ul_snr", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_dl_mcs = np.mean([ue.get("ue_container", {}).get("dl_mcs", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_ul_mcs = np.mean([ue.get("ue_container", {}).get("ul_mcs", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_dl_bitrate = np.mean([ue.get("ue_container", {}).get("dl_bitrate", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_ul_bitrate = np.mean([ue.get("ue_container", {}).get("ul_bitrate", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_dl_bler = np.mean([ue.get("ue_container", {}).get("dl_bler", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_ul_bler = np.mean([ue.get("ue_container", {}).get("ul_bler", 0) for ue in ue_list]) if num_ues > 0 else 0
                    avg_ul_phr = np.mean([ue.get("ue_container", {}).get("ul_phr", 0) for ue in ue_list]) if num_ues > 0 else 0

                    # Append all the extracted features
                    data.append([
                        nof_rach, num_ues, avg_dl_cqi, avg_ul_snr, avg_dl_mcs, avg_ul_mcs,
                        avg_dl_bitrate, avg_ul_bitrate, avg_dl_bler, avg_ul_bler, avg_ul_phr
                    ])

                    # Label: 0 for clean, 1 for jamming
                    if "clean" in file:
                        labels.append(0)
                    else:
                        labels.append(1)
    return data, labels

# Load training and testing data
train_data, train_labels = load_data_from_files(train_files)
test_data, test_labels = load_data_from_files(test_files)

# Convert to DataFrame
columns = [
    "nof_rach", "num_ues", "avg_dl_cqi", "avg_ul_snr", "avg_dl_mcs", "avg_ul_mcs",
    "avg_dl_bitrate", "avg_ul_bitrate", "avg_dl_bler", "avg_ul_bler", "avg_ul_phr"
]

df_train = pd.DataFrame(train_data, columns=columns)
df_test = pd.DataFrame(test_data, columns=columns)

# Normalize features
scaler = MinMaxScaler()
normalized_train_data = scaler.fit_transform(df_train)
normalized_test_data = scaler.transform(df_test)

# Create sequences of 80 timestamps for LSTM input
sequence_length = 80

def create_sequences(data, labels, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(labels[i + sequence_length - 1])  # Use the label corresponding to the last element in the sequence
    return np.array(X), np.array(y)

# Create sequences for training and testing
X_train, y_train = create_sequences(normalized_train_data, train_labels, sequence_length)
X_test, y_test = create_sequences(normalized_test_data, test_labels, sequence_length)

# Define and Train the LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(sequence_length, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

