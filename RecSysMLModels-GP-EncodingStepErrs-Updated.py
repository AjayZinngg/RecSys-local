import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from dask import delayed, compute
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from dask.diagnostics import ProgressBar

# Load the datasets using Dask
train_data = dd.read_csv('train_set.csv')
test_data = dd.read_csv('test_set.csv')

# Sample a subset of the data (optional, adjust frac as needed)
fraction_of_data_to_use = 0.01  # Adjust this value to suit your needs
train_data = train_data.sample(frac=fraction_of_data_to_use)
test_data = test_data.sample(frac=fraction_of_data_to_use)

# Ensure 'checkin' is in string format
train_data['checkin'] = train_data['checkin'].astype(str)
test_data['checkin'] = test_data['checkin'].astype(str)

# Create a new column that combines 'utrip_id' and 'checkin'
train_data['utrip_id_checkin'] = train_data['utrip_id'].astype(str) + '_' + train_data['checkin']
test_data['utrip_id_checkin'] = test_data['utrip_id'].astype(str) + '_' + test_data['checkin']

# Create a city_country column
train_data['city_country'] = train_data['city_id'].astype(str) + '_' + train_data['hotel_country'].astype(str)
test_data['city_country'] = test_data['city_id'].astype(str) + '_' + test_data['hotel_country'].astype(str)

# Handle missing values
train_data['city_country'] = train_data['city_country'].fillna('missing')
test_data['city_country'] = test_data['city_country'].fillna('missing')

# Convert city_country to category type for efficient encoding
train_data = train_data.categorize(columns=['city_country'])
test_data = test_data.categorize(columns=['city_country'])

# Group by utrip_id to create sequences
with ProgressBar():
    train_sequences = train_data.groupby('utrip_id')['city_country'].apply(list).compute().tolist()
    test_sequences = test_data.groupby('utrip_id')['city_country'].apply(list).compute().tolist()

# Encode city_country strings as integers
all_sequences = train_sequences + test_sequences
all_cities_countries = [city_country for seq in all_sequences for city_country in seq]
encoder = LabelEncoder()
encoder.fit(all_cities_countries)

encoded_train_sequences = [encoder.transform(seq).tolist() for seq in train_sequences]
encoded_test_sequences = [encoder.transform(seq).tolist() for seq in test_sequences]

# Prepare data for training models
def prepare_data(sequences, sequence_length=None):
    if sequence_length is None:
        sequence_length = max(len(seq) for seq in sequences)
    X, y = [], []
    for seq in tqdm(sequences, desc="Preparing data"):
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
    X = pad_sequences(X, maxlen=sequence_length, padding='pre')
    y = np.array(y)
    return X, y

X_train, y_train = prepare_data(encoded_train_sequences)
X_test, y_test = prepare_data(encoded_test_sequences, sequence_length=X_train.shape[1])

# Print shapes to verify the data preparation
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Collaborative Filtering (Item-Based)
item_cooccurrence_matrix = np.zeros((len(encoder.classes_), len(encoder.classes_)))

for seq in encoded_train_sequences:
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            item_cooccurrence_matrix[seq[i], seq[j]] += 1
            item_cooccurrence_matrix[seq[j], seq[i]] += 1

# Use TruncatedSVD for dimensionality reduction
embedding_dim = 50
svd = TruncatedSVD(n_components=embedding_dim)
item_embeddings = svd.fit_transform(item_cooccurrence_matrix)

# Calculate cosine similarity matrix for embeddings
item_sim_matrix = cosine_similarity(item_embeddings)

def collaborative_filtering_predict(current_place):
    if current_place in encoder.classes_:
        current_idx = encoder.transform([current_place])[0]
        similarity_scores = item_sim_matrix[current_idx]
        most_similar_idx = similarity_scores.argsort()[-2]
        return encoder.inverse_transform([most_similar_idx])[0]
    else:
        return None

collab_preds = [collaborative_filtering_predict(encoder.inverse_transform([seq[-1]])[0]) for seq in encoded_test_sequences]

# Markov Chains
transitions = []

for chain in encoded_train_sequences:
    for i in range(len(chain) - 1):
        transitions.append((chain[i], chain[i + 1]))

transitions_df = pd.DataFrame(transitions, columns=['current_place', 'next_place'])

transition_counts = transitions_df.groupby('current_place')['next_place'].value_counts(normalize=True).unstack(fill_value=0)

def markov_chain_predict(current_place):
    if current_place in transition_counts.index:
        return transition_counts.loc[current_place].idxmax()
    else:
        return None

markov_preds = [markov_chain_predict(seq[-1]) for seq in encoded_test_sequences]
# markov_preds = [encoder.inverse_transform([pred])[0] if pred is not None else 'unknown' for pred in markov_preds]

# Function to evaluate models
def evaluate_model(y_true, y_pred):
    # Ensure y_pred contains only labels present in encoder.classes_
    y_pred_mapped = []
    for label in y_pred:
        if label in encoder.classes_:
            y_pred_mapped.append(label)
        else:
            # Handle previously unseen labels, e.g., by mapping to a default label
            y_pred_mapped.append('unknown')  # Replace with appropriate handling

    # Filter y_true to only include labels that are in encoder.classes_
    y_true_filtered = [label for label in y_true if label in encoder.classes_]

    # Transform y_true and y_pred_mapped
    y_true_encoded = encoder.transform(y_true_filtered)
    y_pred_encoded = encoder.transform(y_pred_mapped)

    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
    return accuracy, precision

# Random Forest Model
@delayed
def train_rf(X_train, y_train, X_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    return rf_preds

# Gradient Boosting Model
@delayed
def train_gbm(X_train, y_train, X_test):
    gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm_model.fit(X_train, y_train)
    gbm_preds = gbm_model.predict(X_test)
    return gbm_preds

# LSTM Model
@delayed
def train_lstm(X_train, y_train, X_test):
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=len(encoder.classes_), output_dim=50, input_length=X_train.shape[1]))
    lstm_model.add(LSTM(100, return_sequences=False))
    lstm_model.add(Dense(len(encoder.classes_), activation='softmax'))
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    
    # Predict probabilities using softmax output
    lstm_preds = lstm_model.predict(X_test)
    
    # Use np.argmax to get the index of the class with the highest probability
    lstm_preds_idx = np.argmax(lstm_preds, axis=1)
    
    # Convert the predicted indices back to original labels
    lstm_preds_labels = encoder.inverse_transform(lstm_preds_idx)
    
    return lstm_preds_labels


# Train and predict using Dask delayed functions
rf_preds, gbm_preds, lstm_preds = compute(train_rf(X_train, y_train, X_test), train_gbm(X_train, y_train, X_test), train_lstm(X_train, y_train, X_test))

# Evaluate all models
markov_accuracy, markov_precision = evaluate_model(encoder.inverse_transform(y_test), markov_preds)
rf_accuracy, rf_precision = evaluate_model(encoder.inverse_transform(y_test), rf_preds)
gbm_accuracy, gbm_precision = evaluate_model(encoder.inverse_transform(y_test), gbm_preds)
lstm_accuracy, lstm_precision = evaluate_model(encoder.inverse_transform(y_test), lstm_preds)

# Print the results
print(f"Markov Chains - Accuracy: {markov_accuracy:.2f}, Precision: {markov_precision:.2f}")
print(f"Random Forest - Accuracy: {rf_accuracy:.2f}, Precision: {rf_precision:.2f}")
print(f"Gradient Boosting - Accuracy: {gbm_accuracy:.2f}, Precision: {gbm_precision:.2f}")
print(f"LSTM - Accuracy: {lstm_accuracy:.2f}, Precision: {lstm_precision:.2f}")
# Function to save predictions to CSV
def save_predictions(predictions, filename):
    preds_df = pd.DataFrame({'predicted_next_city_country': predictions})
    preds_df.to_csv(filename, index=False)
    print(f'Predictions written to {filename}')

# Save the predictions for each model
model_predictions = {
    'collab_predictions.csv': collab_preds,
    'markov_predictions.csv': markov_preds,
    'rf_predictions.csv': rf_preds,
    'gbm_predictions.csv': gbm_preds,
    'lstm_predictions.csv': lstm_preds
}

for filename, preds in model_predictions.items():
    save_predictions(encoder.inverse_transform(preds), filename)