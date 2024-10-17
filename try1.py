import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data from JSON files
def load_json_data(filepaths):
    data = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            file_data = json.load(file)
            data.extend(file_data)
    return data

# Extract relevant features from the JSON data
def extract_features(data):
    features = []
    labels = []
    for entry in data:
        if entry['type'] == 'metrics':
            cell_data = entry['cell_list'][0]['cell_container']
            rach_failures = cell_data['nof_rach']
            ul_snr = 0
            ul_bler = 0
            if cell_data['ue_list']:
                ue_data = cell_data['ue_list'][0]['ue_container']
                ul_snr = ue_data.get('ul_snr', 0)
                ul_bler = ue_data.get('ul_bler', 0)
            features.append([rach_failures, ul_snr, ul_bler])
            labels.append(1 if 'jammer' in entry else 0)
    return pd.DataFrame(features, columns=['nof_rach', 'ul_snr', 'ul_bler']), pd.Series(labels)

# Preprocess the data
jammer_files = ['/home/manju/Downloads/kpm-data/kpm-data2/jammer1.json', '/home/manju/Downloads/kpm-data/kpm-data2/jammer2.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer3.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer4.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer5.json','/home/manju/Downloads/kpm-data/kpm-data2/jammer6.json']
clean_files = ['/home/manju/Downloads/kpm-data/kpm-data2/clean1.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean2.json', '/home/manju/Downloads/kpm-data/kpm-data2/clean3.json','/home/manju/Downloads/kpm-data/kpm-data2/clean4.json']

jammer_data = load_json_data(jammer_files)
clean_data = load_json_data(clean_files)

# Extract features from both datasets
jammer_features, jammer_labels = extract_features(jammer_data)
clean_features, clean_labels = extract_features(clean_data)

# Combine and label the datasets
X = pd.concat([jammer_features, clean_features], ignore_index=True)
y = pd.concat([jammer_labels, clean_labels], ignore_index=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

