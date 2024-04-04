from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from feature_extraction import extract_features, get_features
from data_preprocessing import load_data

# Load and preprocess data
RAV = '../data/emotions_clips/audio_speech_actors_01-24/'
df = load_data(RAV)
X, Y = get_features(df)

# Model definition
model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile, train, and evaluate model
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_lstm.fit(...)
