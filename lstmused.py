import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load JSON files and concatenate into a dataframe
files = ['/home/manju/Downloads/kpm-data/kpm-data2/jammer1.json', '/home/manju/Downloads/kpm-data/kpm-data2/jammer2.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer3.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer4.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer5.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer6.json','/home/manju/Downloads/kpm-data/kpm-data2/clean1.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean2.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean3.json','/home/manju/Downloads/kpm-data/kpm-data2/clean4.json']
data = []
labels = []

for file in files:
    with open(file, 'r') as f:
        json_data = json.load(f)
        for entry in json_data:
            # Extract relevant features
            if entry["type"] == "metrics":
                carrier_id = entry["cell_list"][0]["cell_container"]["carrier_id"]
                nof_rach = entry["cell_list"][0]["cell_container"]["nof_rach"]
                ue_list = entry["cell_list"][0]["cell_container"]["ue_list"]

                # If there are UEs, aggregate metrics
                num_ues = len(ue_list)
                avg_dl_cqi = np.mean([ue.get("ue_container", {}).get("dl_cqi", 0) for ue in ue_list]) if num_ues > 0 else 0
                avg_ul_snr = np.mean([ue.get("ue_container", {}).get("ul_snr", 0) for ue in ue_list]) if num_ues > 0 else 0

                # Append feature list
                data.append([nof_rach, num_ues, avg_dl_cqi, avg_ul_snr])

                # Label: 0 for clean, 1 for jamming
                if "clean" in file:
                    labels.append(0)
                else:
                    labels.append(1)

# Convert to DataFrame
df = pd.DataFrame(data, columns=["nof_rach", "num_ues", "avg_dl_cqi", "avg_ul_snr"])
labels = np.array(labels)

# Normalize features
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

# Create sequences of 50 timestamps
sequence_length = 50
X = []
y = []

for i in range(len(normalized_data) - sequence_length):
    X.append(normalized_data[i:i + sequence_length])
    y.append(labels[i + sequence_length - 1])  # Use the label corresponding to the last element in the sequence

X = np.array(X)
y = np.array(y)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the LSTM Model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('lstm_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Calculate predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc}")
