from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from feature_extraction import extract_features, get_features
from data_preprocessing import load_data

# Load and preprocess data
RAV = '../data/emotions_clips/audio_speech_actors_01-24/'
df = load_data(RAV)
X, Y = get_features(df)

# Model definition
model = Sequential([
    Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=5, strides=2, padding='same'),
    # Add more layers...
    Flatten(),
    Dense(7, activation='softmax')
])

# Compile, train, and evaluate model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(...)
