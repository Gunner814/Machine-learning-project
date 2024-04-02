# import os
# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import LSTM, Dense, Masking
# from tensorflow.keras.models import Sequential

# # Feature extraction without fixed-length padding
# def extract_features(file_path):
#     try:
#         audio, sample_rate = librosa.load(file_path, sr=None)
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     except Exception as e:
#         print("Error encountered while parsing file: ", file_path, e)
#         return None
#     return mfccs.T  # Transpose to make time steps the first dimension

# # Load dataset and extract features
# def load_data(data_path, audio_dir):
#     data_df = pd.read_csv(data_path, sep='\t')
#     features, labels = [], []
#     for _, row in data_df.iterrows():
#         file_path = os.path.join(audio_dir, row['path'])
#         mfccs = extract_features(file_path)
#         if mfccs is not None:
#             features.append(mfccs)
#             labels.append(row['sentence'])  # Assuming 'sentence' column exists
#     return features, labels

# # Preprocess and tokenize labels
# def preprocess_labels(labels):
#     tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
#     tokenizer.fit_on_texts(labels)
#     sequences = tokenizer.texts_to_sequences(labels)
#     return sequences, tokenizer

# # Create a TensorFlow dataset for variable-length sequences
# def make_dataset(features, sequences, batch_size=32):
#     dataset = tf.data.Dataset.from_generator(
#         lambda: zip(features, sequences),
#         output_signature=(
#             tf.TensorSpec(shape=(None, 40), dtype=tf.float32),
#             tf.TensorSpec(shape=(None,), dtype=tf.int32))
#     )
#     dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 40], [None]))
#     return dataset

# # Define the model
# def build_model(input_dim, output_dim):
#     model = Sequential([
#         Masking(mask_value=0., input_shape=(None, input_dim)),
#         LSTM(128, return_sequences=True),
#         LSTM(64),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Main script
# if __name__ == "__main__":
#     data_path = '../data/validated.tsv'
#     audio_dir = '../data/clips'
    
#     # Load and preprocess data
#     features, labels = load_data(data_path, audio_dir)
#     sequences, tokenizer = preprocess_labels(labels)
    
#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(features, sequences, test_size=0.2, random_state=42)
    
#     # Create TensorFlow datasets
#     train_dataset = make_dataset(X_train, y_train)
#     val_dataset = make_dataset(X_val, y_val)
    
#     # Build and train the model
#     model = build_model(40, len(tokenizer.word_index) + 1)  # 40 MFCC features
#     model.summary()
#     model.fit(train_dataset, validation_data=val_dataset, epochs=10)

###############################################################################################################

# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer

# # 1. Preprocessing
# def extract_mfcc(file_path):
#     audio, sample_rate = librosa.load(file_path, sr=None)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
#     return mfccs.T

# def preprocess_data(dataframe, audio_dir):
#     tokenizer = Tokenizer(char_level=True)
#     tokenizer.fit_on_texts(dataframe['sentence'].values)
#     sequence_data = tokenizer.texts_to_sequences(dataframe['sentence'].values)
#     sequence_padded = pad_sequences(sequence_data, padding='post')
    
#     mfcc_features = []
#     for index, row in dataframe.iterrows():
#         file_path = f"{audio_dir}/{row['path']}"
#         mfccs = extract_mfcc(file_path)
#         mfcc_features.append(mfccs)
    
#     mfcc_padded = pad_sequences(mfcc_features, maxlen=115, dtype='float', padding='post', truncating='post', value=0)
    
#     return mfcc_padded, sequence_padded, tokenizer

# # CTC loss function
# def ctc_lambda_func(args):
#     y_pred, labels, input_length, label_length = args
#     return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

# # Assuming you have a dataframe with 'filename' and 'transcription' columns
# dataframe = data_df = pd.read_csv("../data/validated.tsv", sep='\t')
# audio_dir = '../data/clips'
# mfcc_features, transcriptions, tokenizer = preprocess_data(dataframe, audio_dir)

# # 2. Preparing the dataset
# X_train, X_test, y_train, y_test = train_test_split(mfcc_features, transcriptions, test_size=0.2)
# input_length = np.array([len(x) for x in X_train])
# label_length = np.array([len(y) for y in y_train])

# # 3. Defining the model
# input_dim = X_train.shape[2]  # MFCC feature dimension
# output_dim = len(tokenizer.word_index) + 2  # Num characters + 1 for CTC blank character

# input_data = Input(name='the_input', shape=(None, input_dim), dtype='int64')
# x = Bidirectional(LSTM(128, return_sequences=True))(input_data)
# x = TimeDistributed(Dense(output_dim, activation='linear'))(x)  # linear activation for CTC
# y_pred = Lambda(lambda x: tf.keras.backend.softmax(x, axis=-1))(x)

# labels = Input(name='the_labels', shape=[None], dtype='int64')
# input_lengths = Input(name='input_length', shape=[1], dtype='int64')
# label_lengths = Input(name='label_length', shape=[1], dtype='int64')

# loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_lengths, label_lengths])
# model = Model(inputs=[input_data, labels, input_lengths, label_lengths], outputs=loss_out)

# # 4. Training the model
# model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
# model.fit(x=[X_train, y_train, input_length, label_length], y=np.zeros(len(X_train)), batch_size=32, epochs=10, validation_split=0.2)

##########################################################################################################################################

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
    x = TimeDistributed(Dense(vocab_size + 1, activation='softmax'))(x)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    return model


mfcc_features = []
temp_sequences = []

for index, row in data_df.iterrows():
    file_path = os.path.join(audio_dir, row['path'])
    mfccs = extract_mfcc(file_path)
    if mfccs is not None:
        mfcc_features.append(mfccs.T)  # Transpose to match LSTM input requirements
        temp_sequences.append(row['sentence'])
sequences, tokenizer, max_text_len = preprocess_text(temp_sequences)

#print(len(mfcc_features) + " " + len(sequences) + "\n")
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
    epochs=10,
    batch_size=32
)

#model.save('model/speech_recognition_model.h5')
model.save('model/speech_recognition_model.keras')