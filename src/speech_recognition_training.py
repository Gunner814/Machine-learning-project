import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K

# Assuming 'data_path' is the path to your 'validated.tsv',
# and 'audio_dir' is the directory containing the corresponding audio files.
data_path = '../data/validated.tsv'
audio_dir = '../data/clips'

# Load dataset
data_df = pd.read_csv(data_path, sep='\t')

# Extract MFCCs from audio files
def extract_mfcc(file_path, max_pad_len=400):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        return None
    return mfccs

# Preprocess text labels
def preprocess_text(texts):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return sequences, tokenizer, max_len

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(input_shape, vocab_size):
    input_data = Input(name='input', shape=input_shape)
    labels = Input(name='labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')

    x = Bidirectional(LSTM(128, return_sequences=True))(input_data)
    x = TimeDistributed(Dense(vocab_size + 1, activation='softmax'), name='dense')(x)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    return model

def predict_transcription(model, file_path, tokenizer):
    features = extract_mfcc(file_path, max_pad_len=400)
    if features is None:
        print("Could not extract features for", file_path)
        return ""

    # Adjust the dimensions for model prediction: (batch_size, timesteps, features)
    features = np.expand_dims(features, axis=0)

    # Get the prediction from the model
    pred = model.predict(features)

    # Decode the predicted probabilities
    input_length = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = tf.keras.backend.ctc_decode(pred, input_length=input_length, greedy=True)[0][0]
    
    # Convert decoded sequences to text
    text = ''.join([tokenizer.index_word[p] for p in decoded.numpy()[0] if p != -1])
    
    return text

mfcc_features = []
temp_sequences = []

for index, row in data_df.iterrows():
    file_path = os.path.join(audio_dir, row['path'])
    mfccs = extract_mfcc(file_path)
    if mfccs is not None:
        mfcc_features.append(mfccs.T)  # Transpose to match LSTM input requirements
        temp_sequences.append(row['sentence'])
sequences, tokenizer, max_text_len = preprocess_text(temp_sequences)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(mfcc_features, sequences, test_size=0.2, random_state=42)

# Convert to numpy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Calculate lengths for CTC loss
input_length_train = np.array([X_train.shape[1]] * X_train.shape[0])
input_length_val = np.array([X_val.shape[1]] * X_val.shape[0])
label_length_train = np.array([len(seq) for seq in y_train])
label_length_val = np.array([len(seq) for seq in y_val])

vocab_size = len(tokenizer.word_index) + 1
model = build_model((None, 40), vocab_size)  # (None, 40) is the input shape (time steps, features)
model.summary()

# Prepare data for CTC; convert to the right shapes and types
train_inputs = {
    'input': X_train,
    'labels': y_train,
    'input_length': input_length_train,
    'label_length': label_length_train,
}
train_outputs = {'ctc': np.zeros(len(X_train))}

val_inputs = {
    'input': X_val,
    'labels': y_val,
    'input_length': input_length_val,
    'label_length': label_length_val,
}
val_outputs = {'ctc': np.zeros(len(X_val))}

# Train the model
history = model.fit(
    x=train_inputs,
    y=train_outputs,
    validation_data=(val_inputs, val_outputs),
    epochs=1,
    batch_size=32
)

audio_file_path = '../data/test_clips/test.mp3'  # Update with the path to your audio file
#prediction_model = Model(inputs=model.get_layer('input').input, outputs=model.get_layer('dense').output)
input_data = model.input
prediction_model = Model(inputs=[input_data], outputs=model.get_layer('dense').output)
predicted_text = predict_transcription(prediction_model, audio_file_path, tokenizer)

with open('../data/predicted_transcriptions.txt', 'w') as file:
    file.write(predicted_text)
#model.save('../models/speech_recognition_model.keras')