# import os
# import pandas as pd
# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# # Load metadata
# metadata_path = '../data/validated.tsv'  # Path to your metadata file
# metadata = pd.read_csv(metadata_path, sep='\t')

# # Define a function to load and preprocess audio
# def extract_features(file_path, max_pad_len=40):
#     try:
#         audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#         #pad_width = max_pad_len - mfccs.shape[1]
#         #mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         if mfccs.shape[1] > max_pad_len:
#             mfccs = mfccs[:, :max_pad_len]
#         else:
#             pad_width = max_pad_len - mfccs.shape[1]
#             mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     except Exception as e:
#         print(f"Error encountered while parsing file: {file_path}, {e}")
#         return None 
#     return mfccs

# audio_dir = '../data/clips'  # Directory containing your MP3 files
# features = []
# labels = []

# for index, row in metadata.iterrows():
#     file_path = '../data/clips/' + row['path'] #os.path.join(audio_dir, row['path'])
#     data = extract_features(file_path)
#     if data is not None:
#         features.append(data)
#         labels.append(row['sentence'])  # Update this based on your actual label

# features = np.array(features)
# labels = np.array(labels)  # Ensure this matches your actual labels structure
# labels = to_categorical(labels, num_classes=2)  # Adjust num_classes as needed

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# X_train = X_train[..., np.newaxis]  # Add channel dimension for CNN input
# X_test = X_test[..., np.newaxis]

# # Build a simple CNN model
# model = Sequential([
#     Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(40, 40, 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(2, activation='softmax')  # Update output layer based on your number of classes
# ])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=30, batch_size=256, validation_data=(X_test, y_test), verbose=2)

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

# # Export evaluation results to a text file
# with open('../data/model_evaluation.txt', 'w') as file:
#     file.write(f"Test Loss: {test_loss}\n")
#     file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

# # Plot training & validation accuracy and loss
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from sklearn.model_selection import train_test_split

# Load metadata
metadata_path = '../data/validated.tsv'  # Adjust path as needed
metadata = pd.read_csv(metadata_path, sep='\t')

audio_dir = '../data/clips'  # Adjust directory as needed

# Define a function to load and preprocess audio
def extract_features(file_path, max_pad_len=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None 
    return mfccs

# Preprocess audio and labels
features = []
labels = []

for index, row in metadata.iterrows():
    file_path = os.path.join(audio_dir, row['path'])
    data = extract_features(file_path)
    if data is not None:
        features.append(data)
        labels.append(row['sentence'])  # Assuming 'sentence' contains the transcription

# Tokenize and pad text labels
tokenizer = Tokenizer()
tokenizer.fit_on_texts(labels)
sequences = tokenizer.texts_to_sequences(labels)
padded_sequences = pad_sequences(sequences, padding='post')

# Convert features to NumPy array and prepare data for training
features = np.array(features)
X_train, X_test, y_train, y_test = train_test_split(features, padded_sequences, test_size=0.2, random_state=42)

# Add channel dimension for LSTM layer
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Define a basic LSTM model for demonstration
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(64, activation='relu'),
    LSTM(64, return_sequences=True),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, np.expand_dims(y_train, -1), epochs=10, validation_data=(X_test, np.expand_dims(y_test, -1)), verbose=2)
