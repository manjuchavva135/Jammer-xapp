import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load JSON files and concatenate into a dataframe
files = ['/home/manju/Downloads/kpm-data/kpm-data2/jammer1.json', '/home/manju/Downloads/kpm-data/kpm-data2/jammer2.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer3.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer4.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer5.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer6.json','/home/manju/Downloads/kpm-data/kpm-data2/clean1.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean2.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean3.json','/home/manju/Downloads/kpm-data/kpm-data2/clean4.json']

data = []
labels = []

for file in files:
    with open(file, 'r') as f:
        json_data = json.load(f)
        for entry in json_data:
            if entry["type"] == "metrics":
                nof_rach = entry["cell_list"][0]["cell_container"]["nof_rach"]
                ue_list = entry["cell_list"][0]["cell_container"]["ue_list"]
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

                data.append([
                    nof_rach, num_ues, avg_dl_cqi, avg_ul_snr, avg_dl_mcs, avg_ul_mcs,
                    avg_dl_bitrate, avg_ul_bitrate, avg_dl_bler, avg_ul_bler, avg_ul_phr
                ])

                # Label: 0 for clean, 1 for jamming
                if "clean" in file:
                    labels.append(0)
                else:
                    labels.append(1)

# Prepare DataFrame
columns = ["nof_rach", "num_ues", "avg_dl_cqi", "avg_ul_snr", "avg_dl_mcs", "avg_ul_mcs",
           "avg_dl_bitrate", "avg_ul_bitrate", "avg_dl_bler", "avg_ul_bler", "avg_ul_phr"]
df = pd.DataFrame(data, columns=columns)
labels = np.array(labels)

# Normalize features
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

# Create sequences for LSTM
sequence_length = 80
X = []
y = []

for i in range(len(normalized_data) - sequence_length):
    X.append(normalized_data[i:i + sequence_length])
    y.append(labels[i + sequence_length - 1])

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for the Attack App
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
print("Test data saved as 'X_test.npy' and 'y_test.npy'.")

# Build and train the model
model = Sequential([
    LSTM(128, activation='tanh', input_shape=(sequence_length, X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("lstm_jammer_detection.h5")
print(f"Model saved as 'lstm_jammer_detection.h5'.")
