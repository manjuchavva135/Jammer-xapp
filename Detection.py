import json
import os
import numpy as np
import pandas as pd

# Directory path to your data files
data_dir = "/home/manju/Downloads/kpm-data/kpm-data2"

# List of file names with labels (0: no jammer, 1: jammer present)
file_labels = [
    ("clean1.json", 0),  # No jammer
    ("clean2.json", 0),
    ("clean3.json", 0),
    ("clean4.json", 0),
    ("jammer1.json", 1),  # Jammer present
    ("jammer2.json", 1),
    ("jammer3.json", 1),
    ("jammer4.json", 1),
    ("jammer5.json", 1),
    ("jammer6.json", 1)
]

# Initialize lists to store data and labels
all_data = []
all_labels = []

# Load each file and assign the corresponding label
for filename, label in file_labels:
    with open(os.path.join(data_dir, filename), 'r') as f:
        data = json.load(f)
        all_data.append(data)
        all_labels.append(label)

# Check how many files are loaded
print(f"Loaded {len(all_data)} files with corresponding labels.")


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Initialize encoders and scalers
scaler = MinMaxScaler()
encoder = OneHotEncoder(sparse_output=False)

def extract_features_and_preprocess(data):
    feature_data = []
    rnti_ids = []

    for entry in data:
        # Initialize features with all expected keys
        features = {
            'asn1_length': None,
            'selectedPLMN_Identity': None,
            'rlf_InfoAvailable_r10': None,
            'dl_cqi': None,
            'dl_mcs': None,
            'dl_bitrate': None,
            'ul_snr': None,
            'ul_mcs': None,
            'ul_bitrate': None,
            'ul_phr': None,
            'ul_bsr': None,
            'qci': None
        }

        valid_entry = False

        if entry['type'] == 'rrc_log':
            # Extract features from rrc_log
            features.update({
                'asn1_length': entry['asn1_length'],
                'selectedPLMN_Identity': entry['event_data']['selectedPLMN-Identity'],
                'rlf_InfoAvailable_r10': int(entry['event_data']['nonCriticalExtension']['nonCriticalExtension']['rlf-InfoAvailable-r10'] == 'true')
            })
            rnti_ids.append(entry['event_data']['rnti'])
            valid_entry = True
        elif entry['type'] == 'metrics':
            # Check if the necessary lists are not empty
            if entry['cell_list'] and entry['cell_list'][0]['cell_container']['ue_list']:
                ue_info = entry['cell_list'][0]['cell_container']['ue_list'][0]['ue_container']
                if ue_info['bearer_list'] and ue_info['bearer_list'][0]['bearer_container']:
                    features.update({
                        'dl_cqi': ue_info['dl_cqi'],
                        'dl_mcs': ue_info['dl_mcs'],
                        'dl_bitrate': ue_info['dl_bitrate'],
                        'ul_snr': ue_info['ul_snr'],
                        'ul_mcs': ue_info['ul_mcs'],
                        'ul_bitrate': ue_info['ul_bitrate'],
                        'ul_phr': ue_info['ul_phr'],
                        'ul_bsr': ue_info['ul_bsr'],
                        'qci': ue_info['bearer_list'][0]['bearer_container']['qci']
                    })
                    rnti_ids.append(ue_info['ue_rnti'])
                    valid_entry = True

        # Append the feature set only if the entry is valid
        if valid_entry:
            feature_data.append(features)

    # Convert to a DataFrame
    df = pd.DataFrame(feature_data)

    # One-hot encode categorical fields
    encoder.fit(df[['qci']])
    encoded_qci = encoder.transform(df[['qci']])

    # Add one-hot encoded fields back to the DataFrame
    df_encoded = pd.concat([df.drop(['qci'], axis=1), pd.DataFrame(encoded_qci)], axis=1)

    # Normalize numerical features
    numerical_cols = ['asn1_length', 'dl_cqi', 'dl_mcs', 'dl_bitrate', 'ul_snr', 'ul_mcs', 'ul_bitrate', 'ul_phr', 'ul_bsr']
    scaler.fit(df_encoded[numerical_cols])  # Fit the scaler on the numerical columns
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    # Add RNTI to track sequences by UE
    df_encoded['rnti'] = rnti_ids

    # Group data by RNTI (each group represents a unique UE)
    grouped = df_encoded.groupby('rnti')

    # Create a list of sequences (one per UE)
    seqs = [group.drop(['rnti'], axis=1).values for _, group in grouped]

    # Pad sequences to ensure uniform length for LSTM
    max_sequence_length = 50  # Adjust as necessary
    padded_sequences = pad_sequences(seqs, maxlen=max_sequence_length, padding='post', dtype='float32')

    return padded_sequences

# Initialize lists to store sequences and labels
sequences = []
labels = []

# Preprocess all files and store the sequences and labels
for data, label in zip(all_data, all_labels):
    preprocessed_sequences = extract_features_and_preprocess(data)
    sequences.append(preprocessed_sequences)
    labels.append([label] * len(preprocessed_sequences))  # Assign labels for each sequence

# Flatten the lists
X = np.concatenate(sequences, axis=0)
y = np.concatenate(labels, axis=0)

# Check the shapes of the data
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# Another LSTM layer
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# Output layer for binary classification (jammer present or not)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]}")
